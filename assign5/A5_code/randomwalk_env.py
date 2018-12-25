from rl_glue import BaseEnvironment
import numpy as np

class randomWalkEnvironment(BaseEnvironment):
    def __init__(self):
        """Declare environment variables."""
        pass

    def env_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize environment variables necessary for run.
        """
        self.currentState = None
        self.right = 1000
        self.left = 1

    def env_start(self):
        self.currentState = np.zeros(1) + 500
        return self.currentState


    def env_step(self, action):
        self.currentState[0] += action

        # print(self.currentState[0])

        if self.currentState[0] >= self.right:
            reward = 1
            terminal = True
        elif self.currentState[0] <= self.left:
            reward = -1
            terminal = True
        else:
            reward = 0
            terminal = False
        # print(terminal)              
        return reward, self.currentState, terminal

    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        pass
