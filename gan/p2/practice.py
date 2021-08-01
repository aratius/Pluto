# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb?hl=ja#scrollTo=gl7jcC7TdPTG
# 手書き文字自動生成

import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import PIL
from tensorflow.keras import layers
import time

from IPython import display

sys.path.append("/Users/matsumotoarata/git/ME/Python/deep-learning")
from gan.p2.generator import make_generator_model, generator_loss, generate_and_save_images
from gan.p2.discriminator import make_discriminator_model, discriminator_loss

# 手書き文字読み込み
# train_images.shape: (60000, 28, 28)
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# train_images.shape: (60000, 28, 28, 1)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
train_images = (train_images - 127.5) / 127.5   # -1 ~ 1 の正規化

# 訓練用画像をバッチ＆シャッフル
BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# 贋作者のネットワークモデル(keras.Model?)
generator = make_generator_model()
# ノイズ
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap="gray")
# plt.show() # 表示

# 鑑定者のネットワークモデル
discriminator = make_discriminator_model()
# ジャッジ (正なら本物, 負なら偽物と判断した)
decision = discriminator(generated_image)
# 未学習のとき
# tf.Tensor([[-0.00158945]], shape=(1, 1), dtype=float32)
print(decision)

# 最適化
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# トレーニングが長くなって中断されたりしたときのモデルの保存と復元方法
checkpoint_dir = "./gan/p2/traininig_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
  generator_optimizer=generator_optimizer,
  discriminator_optimizer=discriminator_optimizer,
  generator=generator,
  discriminator=discriminator
)


# トレーニングループを定義
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# 訓練は贋作者がランダムなseedを受け取るところから始まる
# それをもとに画像を作り、鑑定者はそれを（本物か偽物か）クラス分けする
# 両者に対して損失関数の計算
# パラメータ更新
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
  noise = tf.random.normal([BATCH_SIZE, noise_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    # まず贋作者が画像を生成
    generated_images = generator(noise, training=True)

    # 鑑定者はその偽物と、データセットの本物を両方判定する
    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    # それぞれの損失を計算
    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

def train(dataset, epochs):
  for epoch in range(epochs):
    # 開始時間
    start = time.time()

    # バッチごとにトレーニング
    for image_batch in dataset:
      train_step(image_batch)

    # GIF作る
    display.clear_output(wait=True)
    generate_and_save_images(
      generator,
      epoch+1,
      seed
    )

    if(epoch + 1) % 15 == 0:  # 15エポックごとにモデルを保存する
      checkpoint.save(file_prefix = checkpoint_prefix)

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  display.clear_output(wait=True)
  generate_and_save_images(
    generator,
    epochs,
    seed
  )

train(train_dataset, EPOCHS)