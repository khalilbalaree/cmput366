import numpy as np
from rl_glue import BaseAgent


class RandomAgent(BaseAgent): 
    def __init__(self):
        # your agent may need to remember what the action taken was
        self.action = None
        self.arms = None
        self.epslion = None
        self.alpha = None
        self.q_start = None
        self.average = None
        self.c = None
        self.ucb_estimate = None

    def set_greedy(self, epslion, alpha, q_start):
        self.epslion = epslion
        self.alpha = alpha
        self.q_start = q_start
        self.average = False

    def set_ucb(self, c, q_start, average = True):
        self.c = c
        self.q_start = q_start
        self.average = average
        self.ucb_estimate = np.zeros(10)

    def agent_init(self):
        self.time_step = 1
        self.action_count = np.zeros(10)
        self.arms = np.zeros(10) + self.q_start

    def _choose_action(self):
        if self.c is not None: 
            for i in range(0, 9):
                self.ucb_estimate[i] = self.arms[i] + self.c * np.sqrt(np.log(self.time_step) / (self.action_count[i] + 0.0000000001))
            self.action = np.argmax(self.ucb_estimate)

        else:
            rand = np.random.uniform(0,1)
            if rand < self.epslion:
                self.action = np.random.randint(0,10)
            else:
                self.action = np.argmax(self.arms)

        self.action_count[int(self.action)] += 1
        self.time_step += 1
        
        return self.action


    def agent_start(self, state):
        action = self._choose_action()

        return action

    def agent_step(self, reward, state):
        if self.average:
            self.arms[int(self.action)] = self.arms[int(self.action)] + 1.0 / self.action_count[int(self.action)] * (reward - self.arms[int(self.action)])
        else:
            self.arms[int(self.action)] = self.arms[int(self.action)] + self.alpha * (reward - self.arms[int(self.action)])

        self.action = self._choose_action()

        return self.action

    def agent_end(self, reward):
        # random agent doesn't care about reward
        pass
