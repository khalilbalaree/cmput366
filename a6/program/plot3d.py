import os
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

filename = 'heights.npy'

if os.path.exists(filename):
    data = np.load(filename)
    # print(data)

    x = np.arange(-1.2, 0.5, 1.7/50)
    y = np.arange(-0.07, 0.07, 0.14/50)
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-to-go')
    ax.plot_wireframe(x, y, data)
    plt.show()