import numpy as np

class Sigmoid:
  def __init__(self):
    # 順伝播の出力値が逆伝播の計算に必要なので覚えておく変数
    self.out = None

  def forward(self, x):
    out = 1 / (1 + np.exp(-x))
    self.out = out

    return out

  def backward(self, dout):
    dx = dout * (1.0 - self.out) * self.out

    return dx

