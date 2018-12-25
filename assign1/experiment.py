import numpy as np
from rl_glue import RLGlue
from bandit_agent import RandomAgent
from bandit_env import Environment1D
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

"""
Question3 and Bonus question 1
"""

def experiment(epslion, alpha, q_start, c):
    agent = RandomAgent()
    environment = Environment1D()
    rlg = RLGlue(environment, agent)
    
    num_run = 200
    max_steps = 1000
    optimal_count = np.zeros(max_steps)

    if epslion is not None:
        agent.set_greedy(epslion, alpha, q_start)
    else:
        agent.set_ucb(c, q_start)

    for run in range(num_run):
        rlg.rl_init()
        rlg.rl_start()

        for i in range(max_steps):
            if rlg.rl_step()[2] == environment.env_message('get optimal action'):
                optimal_count[i] += 1

    return optimal_count / num_run

def main():
    # Question3
    result1 = experiment(0, 0.1, 5, None)
    result2 = experiment(0.1, 0.1, 0, None)

    #Bonus question
    result_ucb = experiment(None, None, 0, 2)

    # draw the plot
    plt.xlabel('step')
    plt.ylabel('%' + ' optional action')
    plt.ylim((0, 1.0))
    plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0), ('0%', '20%', '40%', '60%', '80%', '100%'))
    plt.plot(result1, label = 'Optimistic, greedy, Q1 = 5, epslion = 0', color = 'blue')
    plt.plot(result2, label = 'Realistic, epslion-greedy, Q1 = 0, epslion = 0.1', color = 'grey')
    plt.plot(result_ucb, label = 'UCB c = 2', color = 'black')
    plt.legend()
    plt.show()

main()



