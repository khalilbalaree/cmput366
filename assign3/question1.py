import numpy as np 
import matplotlib.pyplot as plt 

def valueIteration(ph):
    policy = np.zeros(101)
    statesArray = np.zeros(101)
    Value = np.zeros((101,4))
    policy = np.zeros(101)
    sweep = 0
    loop = True

    while loop:
        delta = 0
        theta = 1e-10
        for state in range(1, 100):
            v = statesArray[state]
            result = []
            actions = np.arange(0, min(state, 100 - state) + 1, 1)
            for action in actions:
                win = state + action
                lose = state - action
                if (win == 100):
                    reward = 1
                else:
                    reward = 0
                expection = ph * (reward + statesArray[win]) + (1 - ph) * statesArray[lose]
                result.append(expection)

            statesArray[state] = np.max(result)
            delta = max(delta, abs(v - statesArray[state]))
            policy[state] = np.argmax(result)

            if sweep < 3:
                Value[state][sweep] = statesArray[state]
            else:
                Value[state][3] = statesArray[state]
        
        sweep += 1

        # print(sweep)

        if delta < theta:
            Value[100] = None
            loop = False

    plt.figure('Value ph=' + str(ph))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.xticks([1,25,50,75,99])
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
    plt.margins(0.05)
    plt.plot(Value)
    plt.figure('Policy ph=' + str(ph))
    plt.xlabel('Capital')
    plt.ylabel('stake')
    plt.margins(0.05)
    plt.plot(policy)
    plt.show()


def main():
    ph1 = 0.40
    ph2 = 0.55
    valueIteration(ph1)
    valueIteration(ph2)

main()
    



