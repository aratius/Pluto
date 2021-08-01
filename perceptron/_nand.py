import numpy as np

def NAND(x1, x2):
  # w1,2 = weight, theta = threshold
  # x1,2 1 or 0 input
  x = np.array([x1, x2])
  w = np.array([-0.5, -0.5])
  b = 0.6  # bias = -theta
  tmp = np.sum(x * w) + b
  if tmp <= 0:
    return 0
  elif tmp > 0:
    return 1


print(NAND(0, 0))  # 1
print(NAND(1, 0))  # 1
print(NAND(0, 1))  # 1
print(NAND(1, 1))  # 0