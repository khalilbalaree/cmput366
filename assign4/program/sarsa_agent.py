from rl_glue import BaseAgent
import numpy as np


class SarsaAgent(BaseAgent):
    def __init__(self):
        """Declare agent variables."""
        self.lastAction = None
        self.lastState = None

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.eplison = 0.1
        self.alpha = 0.5
        #9 actions
        if self.num_action == 9:
            self.actions = [[1,0],[0,1],[-1,0],[0,-1],
                            [1,1],[1,-1],[-1,1],[-1,-1],
                            [0,0]]
        elif self.num_action == 8:
            self.actions = [[1,0],[0,1],[-1,0],[0,-1],
                            [1,1],[1,-1],[-1,1],[-1,-1]]
        elif self.num_action == 4:
            self.actions =[[1,0], [0,1], [-1,0], [0,-1]]
        else:
            exit("ERROR")

        self.Q = np.zeros((10,7,self.num_action))
                                                                                                                  
    def directions(self, num):
        self.num_action = num  

    def agent_start(self, state):
        if np.random.uniform(0,1) < self.eplison:
            action = self.actions[np.random.randint(0, self.num_action)]
        else:
            action = self.actions[np.argmax(self.Q[state[0]][state[1]])]

        self.lastState = state
        self.lastAction = action

        return action

    def agent_step(self, reward, state):
        # print(state)

        if np.random.uniform(0,1) < self.eplison:
            action = self.actions[np.random.randint(0, self.num_action)]
        else:
            action = self.actions[np.argmax(self.Q[state[0]][state[1]])]

        self.Q[self.lastState[0]][self.lastState[1]][self.actions.index(self.lastAction)] += self.alpha * (reward + self.Q[state[0]][state[1]][self.actions.index(action)] - self.Q[self.lastState[0]][self.lastState[1]][self.actions.index(self.lastAction)])

        self.lastState = state
        self.lastAction = action
        
        return action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        return

        
    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'ValueFunction':
            return (np.max(self.Q, axis=1)).tostring()
        else:
            return "I dont know how to respond to this message!!"
