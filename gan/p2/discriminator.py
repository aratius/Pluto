# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb?hl=ja#scrollTo=gl7jcC7TdPTG
import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

def make_discriminator_model():
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]))

  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  return model


# 交差誤差 （それぞれの誤差関数の一部につかう
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# 鑑定者の誤差
def discriminator_loss(real_output, fake_output):
  # 本物かどうかとの誤差 完全に本物なら1の配列になるからそれとの誤差
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  # 偽物かどうかとの誤差 完全に偽物なら０の配列になるからそれとの誤差
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

  # 合計値
  total_loss = real_loss + fake_loss
  return total_loss