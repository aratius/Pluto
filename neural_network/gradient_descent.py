# 勾配法
import numpy as np

# 勾配実装
def numerical_gradient(f, x):
  h = 1e-4 # 0.0001
  grad = np.zeros_like(x) # xと同じ形状の配列を生成

  for idx in range(x.size):
    tmp_val = x[idx]

    # f(x+h)の計算
    x[idx] = tmp_val + h
    fxh1 = f(x)

    # f(x-h)の計算
    x[idx] = tmp_val - h
    fxh2 = f(x)

    # 傾き
    grad[idx] = (fxh1 - fxh2) / (2 * h)
    x[idx] = tmp_val # 値をもとに戻す

  return grad

# 3次元の方程式
# f(x0, x1) = x0**2 + x1**2
def function_2(x):
  return x[0]**2 + x[1]**2

# 勾配降下法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
  x =  init_x

  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad

  return x

# xの初期値, ここから近づけていく
init_x = np.array([3.0, 4.0])
# stap_num回繰り返したあとの最終到達点
reached = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)

print(reached)