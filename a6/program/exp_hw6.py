#!/usr/bin/env python

import numpy as np
from agent_hw6 import Agent
from rl_glue import RLGlue
from env_hw6 import Environment


def question_1():
    # Specify hyper-parameters
    global agent, rlglue 

    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)

    num_episodes = 200
    num_runs = 50
    max_eps_steps = 100000

    steps = np.zeros([num_runs, num_episodes])

    for r in range(num_runs):
        print("run number : ", r)
        rlglue.rl_init()
        for e in range(num_episodes):
            rlglue.rl_episode(max_eps_steps)
            steps[r, e] = rlglue.num_ep_steps()
            # print(steps[r, e])
    np.save('steps', steps)

def question_3():
    #need to set num_episodes = 1000 & num_runs = 1 in question_1() first
    steps = 50
    result = np.zeros([50,50])
    weight = rlglue.rl_agent_message("getWeight")

    for i in range(steps):
        pos = -1.2 + (i * 1.7 / steps)
        for j in range(steps):
            vel = -0.07 + (j * 0.14 / steps)
            values = []
            for a in range(3):
                inds = np.zeros(2048)
                for index in agent.F([pos, vel], a):
                    inds[index] = 1
                values.append(np.dot(weight, inds))
            height = np.max(values)
            result[j][i] = - height
    np.save("heights", result)


if __name__ == "__main__":
    question_1()
    question_3()
    print("Done")
