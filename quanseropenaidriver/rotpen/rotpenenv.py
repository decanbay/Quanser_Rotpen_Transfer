# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 02:36:28 2021

@author: Deniz
"""

import numpy as np
import gym
import time
import math

from gym import spaces
from gym.utils import seeding
from rotpen_simulator import forward_model_euler, forward_model_ode
from rotpen_renderer import RotpenRenderer


MOTOR_VOLTAGE_MAX = 3
ACT_MAX = np.asarray([MOTOR_VOLTAGE_MAX], dtype=np.float64)
# OBS_MAX = [theta, alpha, theta_dot, alpha_dot]
OBS_MAX = np.asarray([np.pi / 2, np.pi, np.inf, np.inf], dtype=np.float64)

class QubeSimulator(object):
    """Simulator that has the same interface as the hardware wrapper."""

    def __init__(
        self, forward_model="ode", frequency=250, integration_steps=1, max_voltage=18.0
    ):
        if isinstance(forward_model, str):
            if forward_model == "ode":
                self._forward_model = forward_model_ode
            elif forward_model == "euler":
                self._forward_model = forward_model_euler
            else:
                raise ValueError(
                    "'forward_model' must be one of ['ode', 'euler'] or a callable."
                )
        elif callable(forward_model):
            self._forward_model = forward_model
        else:
            raise ValueError(
                "'forward_model' must be one of ['ode', 'euler'] or a callable."
            )

        self._dt = 1.0 / frequency
        self._integration_steps = integration_steps
        self._max_voltage = max_voltage
        self.state = (
            np.array([0, 0, 0, 0], dtype=np.float64) + np.random.randn(4) * 0.01
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def step(self, action, led=None):
        action = np.clip(action, -self._max_voltage, self._max_voltage)
        self.state = self._forward_model(
            *self.state, action, self._dt, self._integration_steps
        )
        return self.state

    def reset_up(self):
        self.state = (
            np.array([0, 0, 0, 0], dtype=np.float64) + np.random.randn(4) * 0.01
        )
        return self.state

    def reset_down(self):
        self.state = (
            np.array([0, np.pi, 0, 0], dtype=np.float64) + np.random.randn(4) * 0.01
        )
        return self.state

    def reset_encoders(self):
        pass

    def close(self, type=None, value=None, traceback=None):
        pass




class RotpenBaseEnv(gym.Env):
    super
    









