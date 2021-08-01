import numpy as np

# ReLUレイヤの実装
class Relu:
  def __init__(self):
    # booleanのnumpy配列 0以上かどうか
    self.mask = None

  # 順伝播
  def forward(self, x):
    self.mask = (x <= 0) # boolean 0より小さければTrue
    out = x.copy()
    out[self.mask] = 0

    return out

  # 逆伝播
  def backward(self, dout):
    # 順伝播の際の入力値が0以下は逆伝播（微分結果）も0
    dout[self.mask] = 0
    # それ以外はそのまま通す
    dx = dout

    return dx

x = np.array([[1.0, -2.0], [0.0, 2.2]])

relu = Relu()
relu.forward(x)

# 適当に決めた逆伝播の流れてくる値
d = np.array([[1.0, 1.0], [1.0, 1.0]])
d_x = relu.backward(d)

# [[1. 0.]
#  [0. 1.]]
print(d_x)
