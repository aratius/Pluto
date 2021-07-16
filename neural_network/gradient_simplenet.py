import sys, os
sys.path.append("/Users/matsumotoarata/git/ME/Python/deep-learning")  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
  def __init__(self):
    # 重みパラメータをガウス分布で初期化
    self.W = np.random.randn(2, 3)

  # 予測する
  def predict(self, x):
    return np.dot(x, self.W)

  # 損失度合いを計算
  def loss(self, x, t):
    z = self.predict(x)
    y = softmax(z)
    loss = cross_entropy_error(y, t)

    return loss

# 初期化
net = simpleNet()
# 初期座標
x = np.array([0.6, 0.9])
# 予測する
p = net.predict(x)
# 予測結果
mt = np.argmax(p)

# 正解
t = np.array([0, 0, 1])
# 損失
loss = net.loss(x, t)

# 上記の例では2が正解なので2のときにlossがいちばんすくなくなる
print("" + str(mt) + " : " + str(loss))
