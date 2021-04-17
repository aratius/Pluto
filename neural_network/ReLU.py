import numpy as np
import matplotlib.pylab as plt

def ReLU(x):
  # return x * (x > 0.0)  # x > 0.0 expects 1 or 0
  return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = ReLU(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
