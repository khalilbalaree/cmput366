import matplotlib.pyplot as plt
import numpy as np 
import time
from rl_glue import RLGlue
from randomwalk_env import randomWalkEnvironment
from tabular_agent import TabularAgent
from tile_coding_agent import TileCodingAgent


def power(myList):
    return [ x**2 for x in myList ]

def test(agentName, num_runs, num_episodes, trueValue):
    if agentName == "TabularAgent":
        agent = TabularAgent()
    elif agentName == "TileCodingAgent":
        agent = TileCodingAgent()
    else:
        exit("No",agentName)
    environment = randomWalkEnvironment()
    rlglue = RLGlue(environment, agent)

    # ensure both the agents learn from the same trajectories of experience
    np.random.seed(1488834)
    RUNs = np.zeros((num_runs, int(num_episodes/10)))
    episodeList = np.zeros(int(num_episodes/10))
    print("Run for",agentName)
    for run in range(num_runs):
        startTime0 = time.clock()
        print("#" + str(run), end = ' ')
        rlglue.rl_init()
        RMSEs = np.zeros(int(num_episodes/10))
        for episode in range(num_episodes):
            rlglue.rl_episode()
            thisEpisodeValues = rlglue.rl_agent_message("ValueFunction")
            thisRMSE = np.sqrt(np.mean(power(trueValue - thisEpisodeValues)))   
            # print(thisRMSE)

            # ploting performance every 10 episodes
            if episode % 10 == 0:
                RMSEs[episode//10] = thisRMSE
                if run == 0:
                    episodeList[episode//10] = episode
        
        RUNs[run] = RMSEs
        endTime0 = time.clock()
        print("Running time:",str(round(endTime0-startTime0,4)))

    return np.mean(RUNs, axis=0), episodeList



def main():
    runs = 30
    num_episodes = 2000

    try:
        trueValue = np.load("TrueValueFunction.npy")
    except:
        exit("No TrueValueFunction file...Exit program...")


    startTime0 = time.clock()
    rmsves0, ep0 = test("TabularAgent", runs, num_episodes, trueValue)
    endTime0 = time.clock()
    print("End...\nTotal running time for TabularAgent:",str(round(endTime0-startTime0,4)))
    rmsves1, ep1 = test("TileCodingAgent", runs, num_episodes, trueValue)
    endTime1 = time.clock()
    print("End...\nTotal running time for TileCodingAgent:",str(round(endTime1-endTime0,4)))
    print("Ploting...")

    plt.plot(ep0,rmsves0, label = "TabularAgent")
    plt.plot(ep1,rmsves1, label = "TileCodingAgent")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("RMSVE(Averaged 30 runs)")
    plt.savefig("RandomWalk.png")
    plt.show()

main()