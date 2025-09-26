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