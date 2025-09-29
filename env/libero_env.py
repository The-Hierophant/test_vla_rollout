import sys
sys.path.append("../LIBERO")
from libero.libero import benchmark
from env.libero_utils import get_libero_env, get_libero_dummy_action, normalize_gripper_action, invert_gripper_action, save_rollout_video
import gc
import torch

class LiberoEnv:
    def __init__(self, task_name, task_id, trial_id, is_valid, max_steps, config={"num_steps_wait": 10}):
        self.is_valid = is_valid
        self.max_steps = max_steps

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_name]()
        self.task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        self.initial_state = initial_states[trial_id]
        
        self.env = None

        self.config = config
        try:
            self.env, self.task_description = get_libero_env(self.task, resolution=256)
        except Exception as e:
            print(f"*** env initialization failed, error {e} ***")
            if self.env is not None:
                try:
                    self.env.close()  
                except Exception as e:
                    print(f"error when close the env: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            print("gc collect finish")
        
        self.env.reset()
        self.obs = self.env.set_init_state(self.initial_state)
        
        t = 0
        self.valid_images = []
        while t < self.config["num_steps_wait"]:
            self.obs, _, _, _ = self.env.step(get_libero_dummy_action())
            t += 1
            
        if self.is_valid:
            img = self.obs["agentview_image"][::-1, ::-1]
            self.valid_images.append(img)
        
        self.active = True
        self.complete = False
        self.finish_step = 0
        self.task_file_name = f"{task_name}_task_{task_id}_trial_{trial_id}"

    def get_initial_state(self):
        return {
            'type': 'init',
            'obs': self.obs,
            "task_description": self.task_description,
            'valid_images': self.valid_images.copy(),
            'task_file_name': self.task_file_name,
            'active': self.active,
            'complete': self.complete,
            'finish_step': self.finish_step
        }

    def step(self, action):
        if action is None:
            self.close()
            return {'type': 'terminate'}

        if not self.active:
            raise ValueError("Cannot step in an inactive environment. Please reset the environment.")

        step_images = []
        for i in range(len(action)):
            a = action[i]
            normalized_action = normalize_gripper_action(a, binarize=True)
            inverted_action = invert_gripper_action(normalized_action)
            self.obs, reward, done, info = self.env.step(inverted_action.tolist())
            
            if self.is_valid:
                img = self.obs["agentview_image"][::-1, ::-1]
                step_images.append(img)
            
            self.finish_step += 1
            if done or self.finish_step >= self.max_steps:
                self.active = False
                self.complete = done
                break
        
        output_data = {
            'type': 'step',
            'obs': self.obs,
            'active': self.active,
            'complete': self.complete,
            'finish_step': self.finish_step,
            'valid_images': step_images.copy() if self.is_valid else []
        }
        return output_data

    def close(self):
        if self.env:
            self.env.close()
            self.env = None

import multiprocessing as mp
from collections import defaultdict
from tqdm import trange
# =======================================================================
# 1. 多进程环境封装
# =======================================================================
def worker(conn, env_fn):
    """子进程中运行的函数，增加了初始化握手"""
    # 1. 初始化环境，并通知主进程结果
    try:
        env = env_fn()
        conn.send(('ready', None))  # 发送成功信号
    except Exception as e:
        conn.send(('error', e))     # 发送错误信号，并附带异常信息
        conn.close()
        return  # 初始化失败，子进程退出
    # 2. 进入主循环，等待命令
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == 'get_initial_state':
                initial_state = env.get_initial_state()
                conn.send(initial_state)
            elif cmd == 'step':
                result = env.step(data)
                conn.send(result)
            elif cmd == 'close':
                break  # 退出循环，最终会关闭环境
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        # 确保环境在子进程退出时被关闭
        if 'env' in locals() and hasattr(env, 'close'):
            env.close()
        conn.close()
class SubprocVecEnv:
    """
    一个简单的多进程向量化环境封装。
    """
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        
        self.processes = []
        # 使用 atexit 确保在主进程意外退出时也能尝试关闭子进程
        import atexit
        atexit.register(self.close)
        for i in range(self.num_envs):
            process = mp.Process(target=worker, args=(self.child_conns[i], env_fns[i]), daemon=True)
            self.processes.append(process)
            process.start()
            # 在父进程中关闭子连接句柄
            self.child_conns[i].close()
        # NEW: 等待所有子进程完成初始化
        # ---------------------------------------------
        try:
            results = [conn.recv() for conn in self.parent_conns]
        except EOFError: # 如果子进程在发送信号前就崩溃了
             self.close()
             raise RuntimeError("One of the environment processes terminated prematurely during initialization.")
        # 检查初始化结果
        for i, result in enumerate(results):
            status, data = result
            if status == 'error':
                # 一个环境初始化失败，关闭所有进程并抛出异常
                print(f"Error initializing environment {i}: {data}")
                self.close()
                raise data  # 重新抛出子进程中的异常
        # ---------------------------------------------
        print("All environments initialized successfully.")
        
    def get_initial_states(self):
        """从所有环境中获取初始状态"""
        for conn in self.parent_conns:
            conn.send(('get_initial_state', None))
        
        results = [conn.recv() for conn in self.parent_conns]
        return results
    def step(self, actions, active_env_indices):
        """
        在指定的活跃环境中执行一步，actions 的长度应等于 active_env_indices 的长度。
        """
        # 发送指令
        action_idx = 0
        for i in range(self.num_envs):
            if i in active_env_indices:
                self.parent_conns[i].send(('step', actions[action_idx]))
                action_idx += 1
        
        # 接收结果
        results = [None] * self.num_envs
        for i in active_env_indices:
             results[i] = self.parent_conns[i].recv()
        
        return results
    def close(self):
        """关闭所有环境"""
        for conn in self.parent_conns:
            try:
                conn.send(('close', None))
            except BrokenPipeError:
                pass
        for p in self.processes:
            p.join()