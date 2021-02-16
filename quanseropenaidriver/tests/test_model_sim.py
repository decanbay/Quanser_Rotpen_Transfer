#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 03:17:05 2021

@author: deniz
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from gym_brt.envs import RotpenSwingupEnv


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
    for i in range(t):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rews.append(reward)
        obsrvs.append(obs)
        actions.append(action)
        env.render()
        if done:
            pass
            # obs = env.reset()
    t = np.arange(0, duration, step=d_t)
    if plot:
        fig, ax = plt.subplots(4)
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


env = RotpenSwingupEnv()

model = PPO.load('logs/ppo_swing/best_model.zip', env=env)
test_env(env, model,plot=False)
time.sleep(0.1)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')
