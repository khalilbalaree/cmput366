from rl_glue import RLGlue
from windy_env import WindyEnvironment
from sarsa_agent import SarsaAgent
import numpy as np
import matplotlib.pyplot as plt

environment = WindyEnvironment()
agent = SarsaAgent()
rlglue = RLGlue(environment, agent)

def plot(num):
    max_steps = 8000

    step = 0
    episode = 0
    step_list = []
    episode_list = []

    agent.directions(num)
    rlglue.rl_init()

    while step < max_steps:
        rlglue.rl_episode(max_steps)

        step = rlglue.num_steps()
        episode = rlglue.num_episodes()

        step_list.append(step)
        episode_list.append(episode)

    return step_list, episode_list


def main():
    step_list0, episode_list0 = plot(4)
    step_list1, episode_list1 = plot(8)
    step_list2, episode_list2 = plot(9)

    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.plot(step_list0, episode_list0, label = 'action = 4')
    plt.plot(step_list1, episode_list1, label = 'action = 8')
    plt.plot(step_list2, episode_list2, label = 'action = 9')
    plt.legend()
    plt.show()

main()
