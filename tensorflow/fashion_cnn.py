import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(x_train, t_train), (x_test, t_test) = keras.datasets.fashion_mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
  keras.layers.Flatten(),
  keras.layers.Dense(512, activation="relu"),
  keras.layers.Dropout(0.2),
  keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, t_train, epochs=10)

# ランダムに10こ抽出して確認する -----

# x_testの範囲でランダムな10この数値配列
test_num_array = np.random.randint(0, len(x_test), (10))
# それらを推測
test = model.predict(x_test[test_num_array])
# 推測結果と教師データをplot
for i in range(len(test_num_array)):
  predicted = test[i].argmax()
  plt.subplot(2, 5, i + 1)
  # 推測:教師
  plt.title("p:t => " + str(predicted)+":"+str(t_test[test_num_array[i]]))
  plt.imshow(x_test[test_num_array[i]].reshape(28, 28), cmap=None)
plt.show()
