import numpy as np

# 多次元配列（行列）の作成 （簡単だ）
x = np.array([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]])
for i in x:
  for j in i:
    print(j)