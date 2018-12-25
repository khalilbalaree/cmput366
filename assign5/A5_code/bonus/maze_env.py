from rl_glue import BaseEnvironment
import numpy as np


class DynaMazeEnvironment(BaseEnvironment):
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
        self.lastState = None
        self.ymax = 5
        self.xmax = 8
        self.goal = [8,3]
        self.obstacle = [[2,2],[2,3],[2,4],[5,1],[7,3],[7,4],[7,5]] 

    def env_start(self):
        self.currentState = [0,3]
        self.lastState = self.currentState
        return self.currentState


    def env_step(self, action):
        x_action = action[0]
        y_action = action[1]

        self.currentState = [0,0]
        self.currentState[0] = max(min(self.lastState[0] + x_action, self.xmax), 0)
        self.currentState[1] = max(min(self.lastState[1] + y_action, self.ymax), 0)

        # print(self.lastState, "->", action, "->" , self.currentState)

        if self.currentState in self.obstacle:
            return 0, self.lastState, False      

        else:
            if self.currentState == self.goal:
                reward = 1
                terminal = True
                self.currentState = None
                self.lastState = None
                # print("episode End")
            else:
                reward = 0
                terminal = False
                self.lastState = self.currentState
                                        
        return reward, self.currentState, terminal

    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        pass
