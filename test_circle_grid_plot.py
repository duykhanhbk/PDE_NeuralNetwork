import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

def fun(x, y):
  return x**2 + y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z, color='red')
ax.plot_surface(X, Y, X, color='blue')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show(block=False)
plt.draw()
plt.pause(10)

print('If this shows up then program is not blocked')
