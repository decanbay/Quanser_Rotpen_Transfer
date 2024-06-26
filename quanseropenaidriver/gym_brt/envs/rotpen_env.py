# -*- coding: utf-8 -*-
import numpy as np
from gym import spaces
from gym_brt.envs.rotpen_base_env import RotpenBaseEnv

"""
    Description:
        A pendulum is attached to an un-actuated joint to a horizontal arm,
        which is actuated by a rotary motor. The pendulum begins
        downwards and the goal is flip the pendulum up and then to keep it from
        falling by applying a voltage on the motor which causes a torque on the
        horizontal arm.

    Source:
        This is modified for the Quanser Qube Servo2-USB from the Cart Pole
        problem described by Barto, Sutton, and Anderson, and implemented in
        OpenAI Gym: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        This description is also modified from the description by the OpenAI
        team.

    Observation:
        Type: Box(4)
        Num Observation                   Min         Max
        0   Rotary arm angle (theta)     -90 deg      90 deg
        1   Pendulum angle (alpha)       -180 deg     180 deg
        2   Cart Velocity                -Inf         Inf
        3   Pole Velocity                -Inf         Inf
        Note: the velocities are limited by the physical system.

    Actions:
        Type: Real number (1-D Continuous) (voltage applied to motor)

    Reward:
        r(s_t, a_t) = 1 - (0.8 * abs(alpha) + 0.2 * abs(theta)) / pi

    Starting State:
        Theta = 0 + noise, alpha = pi + noise

    Episode Termination:
        When theta is greater than ±90° or after 2048 steps
"""



class RotpenSwingupEnv(RotpenBaseEnv):
    def _reward(self):
        reward = 1 - (
            (0.8 * np.abs(self._alpha) + 0.2 * np.abs(self._target_angle - self._theta))
            / np.pi
        )
        return max(reward, 0)  # Clip for the follow env case

    def _isdone(self):
        done = False
        done |= self._episode_steps >= self._max_episode_steps
        done |= abs(self._theta) > (90 * np.pi / 180)
        return done

    def reset(self):
        super(RotpenSwingupEnv, self).reset()
        state = self._reset_down()
        return state
    

class RotpenSwingupSparseEnv(RotpenSwingupEnv):
    def _reward(self):
        within_range = True
        within_range &= np.abs(self._alpha) < (1 * np.pi / 180)
        within_range &= np.abs(self._theta) < (1 * np.pi / 180)
        return 1 if within_range else 0
    

    