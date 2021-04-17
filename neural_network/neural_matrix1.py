import numpy as np

X = np.array([1, 2])  # 1, 2
W = np.array([[1, 3, 5], [2, 4, 6]])  # 2, 3

Y = np.dot(X, W)
print(Y)