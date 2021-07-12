
import numpy as np

# ２乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

# 交差エントロピー誤差
def cross_entropy_error(y, t):
    # log(0)に鳴らないための微小な値 (infになってしまう)
    delta = 1e-7
    return - np.sum(t * np.log(y + delta))

# 正解は２です (one_hot expression)
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 2が正解だと導き出した場合
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# ８が正解だと導き出した場合
y2 = [0.1, 0.05, 0.0, 0.0, 0.05, 0.1, 0.0, 0.1, 0.6, 0.0]

# 損失度合い（正解しているので少ない
l1 = mean_squared_error(np.array(y1), np.array(t))
# 損失度合い（不正解なので大きい
l2 = mean_squared_error(np.array(y2), np.array(t))

print(l1)
print(l2)

# 損失度合い（正解しているので少ない
l3 = cross_entropy_error(np.array(y1), np.array(t))
# 損失度合い（不正解なので大きい
l4 = cross_entropy_error(np.array(y2), np.array(t))

print(l3)
print(l4)