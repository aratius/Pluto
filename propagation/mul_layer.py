
# 加算レイヤ（ノード）
class MulLayer:
  def __init__(self):
    self.x = None
    self.y = None

  # 順伝播
  def forward(self, x, y):
    self.x = x
    self.y = y
    # 普通の乗算
    out = x * y

    return out

  # 逆伝播
  # dout -> 伝わってきた信号の値
  def backward(self, dout):
    # xとyがひっくり返る
    dx = dout * self.y
    dy = dout * self.x

    return dx, dy


apple = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
# 順伝播
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

# backward
# 逆伝播
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

