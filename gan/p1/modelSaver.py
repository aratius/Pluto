import tensorflow as tf
from tensorflow import keras
import os, sys
sys.path.append("/Users/matsumotoarata/git/ME/Python/deep-learning")

# モデルを保存する
class ModelSaver(keras.callbacks.Callback):
    def __init__(self, gen_model, disc_model):
        self.gen_model = gen_model
        self.disc_model = disc_model

    def on_epoch_end(self, epoch, logs=None):
      # モデルの途中経過を保存
      self.gen_model.save("gan/memod/gen/gen_epoch" + str(epoch) + ".h5")
      self.disc_model.save("gan/memod/disc/disc_epoch" + str(epoch) + ".h5")