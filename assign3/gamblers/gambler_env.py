"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018,
  University of Alberta.
  Gambler's problem environment using RLGlue.
"""
from rl_glue import BaseEnvironment
import numpy as np


class GamblerEnvironment(BaseEnvironment):
    """
    Slightly modified Gambler environment -- Example 4.3 from
    RL book (2nd edition)

    Note: inherit from BaseEnvironment to be sure that your Agent class implements
    the entire BaseEnvironment interface
    """

    def __init__(self):
        """Declare environment variables."""
        pass

    def env_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize environment variables necessary for run.
        """
        self.prob = 0.55
        self.currentState = None

    def env_start(self):
        """
        Arguments: Nothing
        Returns: state - numpy array
        Hint: Sample the starting state necessary for exploring starts and return.
        """
        self.currentState = np.zeros(1) + np.random.randint(99) + 1
        return self.currentState


    def env_step(self, action):
        """
        Arguments: action - integer
        Returns: reward - float, state - numpy array - terminal - boolean
        Hint: Take a step in the environment based on dynamics; also checking for action validity in
        state may help handle any rogue agents.
        """
        terminal = False

        if action > min(self.currentState[0], 100 - self.currentState[0]) or action == 0:
            print('invalid action'+ str(action)+ ' on state ' + str(self.currentState[0]))

        if np.random.uniform(0,1) < self.prob:
            self.currentState[0] += action
        else:
            self.currentState[0] -= action

        if self.currentState[0] == 100:
            reward = 1
            terminal = True
            self.currentState = None
        elif self.currentState[0] == 0:
            reward = 0
            terminal = True  
            self.currentState = None
        else:
            reward = 0

        return reward, self.currentState, terminal

    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        pass
