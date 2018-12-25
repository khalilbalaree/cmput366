from rl_glue import BaseAgent
from tile3 import *
import numpy as np


class Agent(BaseAgent):
    def __init__(self):
        """Declare agent variables."""
        pass

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.alpha = 0.1/8
        self.gamma = 1
        self.epsilon = 0
        self.lamda = 0.9
        self.iht = IHT(2048)
        self.weight = np.random.uniform(-0.001, 0, 2048)
        # reverse, zero throttle, forward
        self.actions = [0, 1, 2]
        self.Z = None
              
    def agent_start(self, state):
        action = self.getAction(state) 
        self.lastAction = action
        self.lastState = state
        self.Z = np.zeros(2048)

        return action


    def agent_step(self, reward, state):
        for i in self.F(self.lastState, self.lastAction):
            reward -= self.weight[i]
            # replacing traces
            self.Z[i] = 1

        action = self.getAction(state)

        for i in self.F(state, action):
            reward += self.gamma * self.weight[i]

        self.weight += self.alpha * reward * self.Z
        self.Z *= self.gamma * self.lamda

        self.lastState = state
        self.lastAction = action

        return action
   

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        for i in self.F(self.lastState, self.lastAction):
            reward -= self.weight[i]
            # replacing traces
            self.Z[i] = 1

        self.weight += self.alpha * reward * self.Z


    def getAction(self,state):
        if np.random.uniform(0,1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            value = []
            for action in self.actions:
                temp = np.zeros(2048)
                for i in self.F(state, action):
                    temp[i] = 1
                value.append(np.dot(self.weight, temp))
            return np.argmax(value)
                
    # returning the set of (indices of) active features for s, a
    def F(self, state, action):
        return tiles(self.iht, 8, [state[0] * 8 / (1.2 + 0.5), state[1] * 8 / (0.07 + 0.07)], [action])
    
    def agent_message(self, in_message):
        if in_message == 'getWeight':
            return self.weight
        else:
            return "I dont know how to respond to this message!!"