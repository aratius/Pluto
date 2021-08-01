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

f = lamda w: net.loss(x, t)

# NOTE: わけわからんくなったのでメモ -----------
# function = 損失関数になっている
# 以前は f(x0, x1) = x0**2 + x1**2 の値を減らすために、ランダムに選んだ点からの傾きを求めていた
# ニューラルネットワークの学習は損失関数の値を減らすことなので、fを損失関数にして、その値が減る傾きを求めることが学習につながる
# 初期位置は初期重みパラメータで、その重みパラメータが少しずつ改善されていく（重みパラメータを上書きする

# 勾配による重みパラメータの変化 deltaW
# 勾配計算は多次元配列に対応した形に修正されている
deltaW = numerical_gradient(f, net.W)

print(net.W)
print(deltaW)

# 上記の例では2が正解なので2のときにlossがいちばんすくなくなる
print("" + str(mt) + " : " + str(loss))