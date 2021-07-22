
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

# 手書き数字データの読み込み
digits = datasets.load_digits()

# 1797, 64
# 8x8=64pxの画像が1797枚ある
# print(digits.data.shape)

# データの数
n = len(digits.data)

# 画像と正解値の表示
# images = digits.images
# labels = digits.target
# for i in range(10):
#   plt.subplot(2, 5, i + 1)
#   plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
#   plt.axis("off")
#   plt.title("Training: " + str(labels[i]))

# plt.show()

# サポートベクターマシン
# gammna: 一つの訓練データが与えるレ影響の大きさ 学習率？
# C: 誤認識の許容度
clf = svm.SVC(gamma=0.001, C=100.0)
# サポートベクターマシンによる訓練 訓練: 6割 テスト: 4割
# data: 入力データ, target: 教師データ
# スラッシュ一つだとfloatになってtype error なる n*6/10 -> n*6//10（切り捨て）
clf.fit(digits.data[:n*6//10], digits.target[:n*6//10])

# 後ろから10個の正解
# print(digits.target[-30:])

# 予測を行う
# print(clf.predict(digits.data[-30:]))

# テストデータの正解　
expected = digits.target[-n*4//10:]
# テストデータの予測
predicted = clf.predict(digits.data[-n*4//10:])
# 正解率
print(metrics.classification_report(expected, predicted))
# 誤認識のマトリックス
print(metrics.confusion_matrix(expected, predicted))

# 結果のプロット
images = digits.images[-n*4//10:]
for i in range(12):
  plt.subplot(3, 4, i+1)
  plt.axis("off")
  plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
  plt.title("Guess: " + str(predicted[i]))

plt.show()