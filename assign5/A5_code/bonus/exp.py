from rl_glue import RLGlue
from dyna_agent import DynaAgent
from maze_env import DynaMazeEnvironment
import numpy as np
import matplotlib.pyplot as plt

environment = DynaMazeEnvironment()
agent = DynaAgent()
rlglue = RLGlue(environment, agent)

def plot(num):
    num_runs = 10
    num_episode = 50
    result = np.zeros((num_runs, num_episode))

    agent.setN(num)
    np.random.seed(1488834)
    print("run for n = ",str(num))

    for run in range(num_runs):
        print("num of run: ", str(run))
        rlglue.rl_init()
        for ep in range(num_episode):
            rlglue.rl_episode()
            result[run][ep]= rlglue.num_ep_steps()
       
    return np.mean(result, axis=0)


def main():
    result1 = plot(0)
    result2 = plot(5)
    result3 = plot(50)

    result1[0] = None
    result2[0] = None
    result3[0] = None

    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode')
    plt.plot(result1, label = 'n = 0')
    plt.plot(result2, label = 'n = 5')
    plt.plot(result3, label = 'n = 50')
    plt.legend()
    plt.savefig("DynaMaze.png")
    plt.show()

main()
