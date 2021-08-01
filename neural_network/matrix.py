import numpy as np

# multiplicative
A = np.array([[1, 2], [3, 4], [5, 6]])  # 2, 3
B = np.array([[1, 2, 3], [4, 5, 6]])  # 3, 2

result = np.dot(A, B)
print(result)