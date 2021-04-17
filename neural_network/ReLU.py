import numpy as np
import matplotlib.pylab as plt

def ReLU(x):
  return x * (x > 0.0)

x = np.arange(-5.0, 5.0, 0.1)
y = ReLU(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
