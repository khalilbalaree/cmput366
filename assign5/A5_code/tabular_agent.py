from rl_glue import BaseAgent
import numpy as np


class TabularAgent(BaseAgent):
    def __init__(self):
        """Declare agent variables."""
        self.weight = None

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.memoryVector = {}
        self.alpha = 0.5
        self.num_states = 1001
        self.weight = np.zeros(self.num_states)
        self.lastFeature = None

    def agent_start(self, state):
        self.lastFeature = self.getFeatureVecter(state[0])
        return self.getAction()

    def agent_step(self, reward, state):
        thisFeature = self.getFeatureVecter(state[0])
     
        # gradient equals to feature vector
        self.weight += self.alpha * (reward + np.dot(self.weight, thisFeature) - np.dot(self.weight, self.lastFeature)) * self.lastFeature
     
        action = self.getAction()
        # print(action)
        self.lastFeature = thisFeature

        return action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        self.weight += self.alpha * (reward + 0 - np.dot(self.weight, self.lastFeature)) * self.lastFeature
        # print("episode End!")
        return


    def getAction(self):
        pos = np.random.uniform(0,1)
        # forward
        if pos < 0.5:
            action = np.random.randint(1,101)
        # backward
        else:
            action = 0 - np.random.randint(1,101)
        # print(action)
        return action

    def getFeatureVecter(self,state):
        if state in self.memoryVector:
            return self.memoryVector[int(state)]
        else:
            vecter = np.zeros(self.num_states)
            vecter[int(state)] = 1
            self.memoryVector[state] = vecter
            return vecter

      
    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'ValueFunction':
            trueValue = np.zeros(self.num_states)
            for s in range(0, self.num_states):
                trueValue[s] = np.dot(self.weight, self.getFeatureVecter(s))
            return trueValue
        else:
            return "I dont know how to respond to this message!!"
