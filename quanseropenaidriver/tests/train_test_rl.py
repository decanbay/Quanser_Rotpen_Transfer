#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 00:37:27 2021

@author: deniz
"""
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from typing import Callable
from gym_brt.envs import RotpenSwingupEnv

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = RotpenSwingupEnv()
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


def test_env(env, model, plot=True):
    '''
    Parameters
    ----------
    env : gym environment
    model : Model
    plot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    '''
    d_t = 1/env._frequency
    duration = 10
    t = int(duration/d_t)
    obsrvs = []
    rews = []
    obs = env.reset()
    actions = []
    s = time.time()
    for i in range(t):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rews.append(reward)
        obsrvs.append(obs)
        actions.append(action)
        env.render()
        if done:
            obs = env.reset()
    e = time.time() - s
    print(e)
    t = np.arange(0, duration, step=d_t)
    if plot:
        _, ax = plt.subplots(4)
        theta = [obs[0] for obs in obsrvs]
        alpha = [obs[1] for obs in obsrvs]
        ax[0].plot(t, theta, 'b')
        ax[1].plot(t, alpha, 'r')
        ax[2].plot(t, rews, 'g')
        ax[3].plot(t, actions, 'y')
        ax[0].set_title('Theta')
        ax[1].set_title('Alpha- pendulum angle')
        ax[2].set_title('Reward')
        ax[3].set_title('Actions')


if __name__ == '__main__':
    max_time_steps = 5e5
    num_cpu = 12

    env = SubprocVecEnv([make_env('env', i) for i in range(num_cpu)])

    eval_env = RotpenSwingupEnv()
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path='./logs/ppo_swing_neg/',
                                 log_path='./logs/ppo_swing_neg/',
                                 eval_freq=int(2500),
                                 deterministic=True, render=False)

    # model = PPO('MlpPolicy', env, verbose=1, device='cpu',
    #             tensorboard_log="./ppo_swing_tensorboard/")
    model = PPO.load('logs/ppo_swing/best_model.zip', env=env)


    s = time.time()
    model.learn(total_timesteps=max_time_steps,
                tb_log_name="ppo_swing_neg",
                callback=eval_callback)

    e = time.time()-s
    print('Took {} seconds to train for {} steps'.format(e, max_time_steps))
    env_test = RotpenSwingupEnv()
    test_env(env_test, model)
    time.sleep(1)
    mean_reward, std_reward = evaluate_policy(model, env_test,
                                              n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')
