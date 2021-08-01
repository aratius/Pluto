import sys, os
sys.path.append("/Users/matsumotoarata/git/ME/Python/deep-learning")  # 親ディレクトリのファイルをインポートするための設定
from neural_network.softmax import softmax
from common.functions import cross_entropy_error
import numpy as np

# softmax-with-lossレイヤ
class SoftmaxWithLoss:
  def __init__(self):
    self.loss = None
    self.y = None
    self.t = None

  # 順伝播
  def forward(self, x, t):
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy_error(self.y, self.t)

    return self.loss

  # 出力層なのでdout=1と初期値を設定できる
  # 流す値は y - t (正解ラベルとの誤差)になる
  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    # データ一個あたりの誤差
    dx = (self.y - self.t) / batch_size

    return dx

