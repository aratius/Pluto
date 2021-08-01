# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

# ステップ関数
def step_function(x):
  return np.array(x > 0, dtype=np.int)

# -5 ~ 5 で 0.1 刻みのnumpy配列
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # define y axis limit
plt.show()