# -*- coding: utf-8 -*-

import numpy as np
import gym
from gym_brt.envs import RotpenSwingupEnv,RotpenSwingupSparseEnv
from stable_baselines3 import PPO, A2C
import time
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env



# env = RotpenSwingupSparseEnv()

from typing import Callable

def make_env(rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = RotpenSwingupSparseEnv()
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    num_cpu = 11  # Number of processes to use
    # Create the vectorized environment
    envs = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # model = A2C('MlpPolicy', env, verbose=0)

    model = PPO('MlpPolicy', envs, verbose=1)
    s_time = time.time()
    total_timesteps=num_cpu*2048*50
    model.learn(total_timesteps=total_timesteps)
    t_time = time.time()-s_time
    print('Totoal time passed = {} seconds'.format(t_time))
    print('{} it/sec'.format(total_timesteps/t_time))
    time.sleep(1)

    model.save("rotpen_sparse_ppo")

    time.sleep(1)
    test_env = RotpenSwingupSparseEnv()
    obs = test_env.reset()
    for i in range(2500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        if i%250==0:
            print(i)
        if done:
            print('Done')
            obs = test_env.reset()
    test_env.close()