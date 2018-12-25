from rl_glue import BaseEnvironment
import numpy as np


class Environment1D(BaseEnvironment):
    def __init__(self):
        # state we always start in
        self.startState = None

        # state we are in currently
        self.currentState = None

        self.arms_count = None
        self.rand_list = None

    def env_init(self):
        self.arms_count = 10
        self.startState = np.zeros(self.arms_count)
        self.rand_list = np.random.normal(0,1,10)


    def env_start(self):
        self.currentState = self.startState
        return self.currentState

    def env_step(self, action):
        reward = np.random.normal(self.rand_list[int(action)], 1.0)
        terminal = False

        return reward, self.currentState, terminal

    def env_message(self, message):
        if message == 'get optimal action':
            return int(np.argmax(self.rand_list))


