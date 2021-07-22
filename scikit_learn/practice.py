from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()

n = len(iris.data)
# print(n)

# 線形サポートベクターマシン
clf = svm.LinearSVC()
# サポートベクターマシンによる訓練
clf.fit(iris.data, iris.target)

# 品種を判定（予測
print(clf.predict([[5.1, 3.5, 1.4, 0.1], [6.5, 2.5, 4.4, 1.4], [5.9, 3.0, 5.2, 1.5]]))
