import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt

# バッチサイズ
batch_size = 128
# 分類クラス数
num_class = 10
# エポック数
epochs = 20

# mnist読み込み
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 正規化
x_train, x_test = x_train / 255.0, x_test / 255.0

# モデルを定義 ニューラルネットワークの設計
model = keras.models.Sequential([
  # 一次元配列にする
  keras.layers.Flatten(),
  # 一層 512のノード 活性化関数はReLU
  keras.layers.Dense(512, activation="relu"),
  # 20%のノードをランダムに消去　（偏りをなくすため
  keras.layers.Dropout(0.2),
  # 出力層 10クラスの出力 出力層の活性化関数
  keras.layers.Dense(10, activation="softmax")
])

# モデルの最適化 最適化手法: こんかいはadam, 損失関数: こんかいはクロス誤差, metrics=[accuracy]: 精度を記録する
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# 学習の実行
model.fit(x_train, y_train, epochs=5)

# まだ見たことのないテストデータを読ませて計測
model.evaluate(x_test, y_test)