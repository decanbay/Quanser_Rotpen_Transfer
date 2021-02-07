# -*- coding: utf-8 -*-

import numpy as np
import gym
from gym_brt.envs import RotpenSwingupEnv,RotpenSwingupSparseEnv
from stable_baselines3 import PPO
import time
from stable_baselines3.common.evaluation import evaluate_policy





if __name__ == '__main__':
    env = RotpenSwingupSparseEnv()
    model = PPO.load("rotpen_sparse_ppo")
    print('Model loaded successfully!')
    time.sleep(1)
    obs = env.reset()
    for i in range(25000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if i%250==0:
            print(i)
        if done:
            print('Done')
            obs = env.reset()
    env.close()