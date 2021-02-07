import gym
from gym_brt.envs import QubeSwingupEnv


num_episodes = 10
num_steps = 250

with QubeSwingupEnv() as env:
    for episode in range(num_episodes):
        state = env.reset()
        for step in range(num_steps):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)