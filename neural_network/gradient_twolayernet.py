import sys, os
sys.path.append("/Users/matsumotoarata/git/ME/Python/deep-learning")  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:

  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    # 重みの初期化
    # ガウス分布でランダムに
    self.params = {}
    # 一層目の重みパラメータ
    self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
    # 一層目のバイアス
    self.params["b1"] = np.zeros(hidden_size)
    # 二層目の重みパラメータ
    self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
    # 二層目のバイアス
    self.params["b2"] = np.zeros(output_size)

  # 予測
  def predict(self, x):
    W1, W2 = self.params["W1"], self.params["W2"]
    b1, b2 = self.params["b1"], self.params["b2"]

    # 一層目の計算まとめて
    a1 = np.dot(x, W1) + b1
    # シグモイド関数（活性化関数）
    z1 = sigmoid(a1)
    # 二層目の計算まとめて
    a2 = np.dot(z1, W2) + b2
    # ソフトマックス関数（出力層の活性化関数）
    y = softmax(a2)

    # 最終出力
    return y

  # 損失度合い
  # x: 入力データ, t: 教師データ
  def loss(self, x, t):
    y = self.predict(x)

    # 予測結果を損失関数に突っ込んで損失度合いを取得する
    return cross_entropy_error(y, t)

  # 正確さ
  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    accuracy = np.sum(y==t) / float(x.shape[0])

    return accuracy

  # 数値勾配
  def numerical_gradient(self, x, t):
    loss_W = lambda W: self.loss(x, t)

    grads = {}

    # 重みとバイアスの各値を勾配降下で学習させてそのパラメーターを更新する
    # 損失関数を微分にかける
    grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
    grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
    grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
    grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

  # 誤差逆伝播法
  # まだ勉強してないところ
  def gradient(self, x, t):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']
    grads = {}

    batch_num = x.shape[0]

    # forward
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    # backward
    dy = (y - t) / batch_num
    grads['W2'] = np.dot(z1.T, dy)
    grads['b2'] = np.sum(dy, axis=0)

    dz1 = np.dot(dy, W2.T)
    da1 = sigmoid_grad(a1) * dz1
    grads['W1'] = np.dot(x.T, da1)
    grads['b1'] = np.sum(da1, axis=0)


    return grads