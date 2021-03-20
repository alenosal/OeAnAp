import math
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def mccormickFunction(x):
    return math.sin(x[0] + x[1]) + (x[0] + x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1

def functionPlot():
    samples = 500
    sampled = np.linspace(5, -5, samples).astype(int)
    x, y = np.meshgrid(sampled, sampled)
    z = np.zeros((len(sampled), len(sampled)))
    for i in range(len(sampled)):
        for j in range(len(sampled)):
            z[i, j] = mccormickFunction(np.array([x[i][j], y[i][j]]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')
    ax.view_init(10, 50)
    plt.show()

functionPlot()
