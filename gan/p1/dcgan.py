# https://keras.io/examples/generative/dcgan_overriding_train_step/
# 顔画像自動生成

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import gdown  # google driveものをDLするパッケージ
from zipfile import ZipFile
sys.path.append("/Users/matsumotoarata/git/ME/Python/deep-learning")
from gan.p1.GAN import GAN
from gan.p1.GANMonitor import GANMonitor
from gan.p1.modelSaver import ModelSaver

# 最初に一回実行したらあとはいらない記述 ----------
# celebAデータセット（顔画像）を読み込み -----
# ディレクトリを作る
# os.makedirs("celeba_gan")

# url  = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
# output = "celeba_gan/data.zip"
# gdown.download(url, output, quiet=True)

# r: 読み取り
# with ZipFile("celeba_gan/data.zip", "r") as zipobj:
#   zipobj.extractall("celeba_gan")

# 最初に一回実行したらあとはいらない記述 ----------

# データセットを作成、値を正規化
dataset = keras.preprocessing.image_dataset_from_directory(
  "gan/me/dataimg", label_mode=None, image_size=(128,128), batch_size=32
)
dataset = dataset.map(lambda x: x / 255.0)

# サンプル画像読み込み・表示
# for x in dataset:
#   plt.axis("off")
#   plt.imshow((x.numpy() * 255).astype("int32")[0])
#   break
# plt.show()

# 鑑定者のニューラルネットワーク
discriminator = keras.Sequential(
  [
    # 64x64px, rgb 3channel
    keras.Input(shape=(128, 128, 3)),
    # 畳み込み層
    layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
    # 活性化関数
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Flatten(),
    # 捨てる
    layers.Dropout(0.2),
    # 全結合層 出力
    layers.Dense(1, activation="sigmoid")
  ],
  name="discriminator"
)
discriminator.summary()

# 贋作者のニューラルネットワーク
latent_dim = 128
generator= keras.Sequential(
  [
    keras.Input(shape=(latent_dim,)),
    layers.Dense(16 * 16 * 128),
    layers.Reshape((16, 16, 128)),
    layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")
  ],
  name="generator"
)
generator.summary()

epochs = 300  # In practice, use ~100 epochs
# GANをインスタンス化
gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
# 最適化手法を設定
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

generator.save("gan/memod/gen/test.h5")
discriminator.save("gan/memod/disc/test.h5")

# 学習
gan.fit(
    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim), ModelSaver(generator, discriminator)]
)

generator.save("gan/memod/gen/last.h5")
discriminator.save("gan/memod/disc/last.h5")