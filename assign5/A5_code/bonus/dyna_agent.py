from rl_glue import BaseAgent
import numpy as np
import random


class DynaAgent(BaseAgent):
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
        self.alpha = 0.1
        self.gamma = 0.95
        #4 actions
        self.actions =[[1,0], [0,1], [-1,0], [0,-1]]
        self.modelState0 = np.zeros((9,6,4))
        self.modelState1 = np.zeros((9,6,4))
        self.modelR = np.zeros((9,6,4))
        self.Q = np.zeros((9,6,4))
        self.observedState = {}
                                                                                                                  
    def setN(self, n):
        self.n = n

    def agent_start(self, state):
        if np.random.uniform(0,1) < self.eplison:
            action = self.actions[np.random.randint(0, 4)]
        else:
            action = self.actions[np.argmax(self.Q[state[0]][state[1]])]

        self.lastState = state
        self.lastAction = action

        return action

    def agent_step(self, reward, state):
        # print(state)

        if np.random.uniform(0,1) < self.eplison:
            action = self.actions[np.random.randint(0, 4)]
        else:
            action = self.actions[np.argmax(self.Q[state[0]][state[1]])]

        # observation
        # reference: https://stackoverflow.com/questions/20585920/how-to-add-multiple-values-to-a-dictionary-key-in-python
        key = (self.lastState[0], self.lastState[1])
        self.observedState.setdefault(key,[])
        self.observedState[key].append(self.actions.index(self.lastAction))

        #Q_learning update
        self.Q[self.lastState[0]][self.lastState[1]][self.actions.index(self.lastAction)] += self.alpha * (reward + self.gamma * max(self.Q[state[0]][state[1]]) - self.Q[self.lastState[0]][self.lastState[1]][self.actions.index(self.lastAction)])
        
        # print(str(reward), str(state[0]), str(state[1]))
        #model learning
        self.modelState0[self.lastState[0]][self.lastState[1]][self.actions.index(self.lastAction)] = state[0]
        self.modelState1[self.lastState[0]][self.lastState[1]][self.actions.index(self.lastAction)] = state[1]
        self.modelR[self.lastState[0]][self.lastState[1]][self.actions.index(self.lastAction)] = reward


        # planning
        for num in range(self.n):
            planS = random.choice(list(self.observedState.keys()))
            # print(self.observedState[planS])
            planA = random.choice(list(self.observedState[planS]))

            planR = int(self.modelR[planS[0]][planS[1]][planA])
            planNextS0= int(self.modelState0[planS[0]][planS[1]][planA])
            planNextS1 = int(self.modelState1[planS[0]][planS[1]][planA])

            self.Q[planS[0]][planS[1]][planA] += self.alpha * (planR + self.gamma * max(self.Q[planNextS0][planNextS1]) - self.Q[planS[0]][planS[1]][planA])
            

        self.lastState = state
        self.lastAction = action
        
        return action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        self.Q[self.lastState[0]][self.lastState[1]][self.actions.index(self.lastAction)] += self.alpha * (reward - self.Q[self.lastState[0]][self.lastState[1]][self.actions.index(self.lastAction)])
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
