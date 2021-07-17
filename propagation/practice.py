import sys, os
sys.path.append("/Users/matsumotoarata/git/ME/Python/deep-learning")  # 親ディレクトリのファイルをインポートするための設定
from propagation.addLayer import AddLayer
from propagation.mullayer import MulLayer

# 計算グラフの演習

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 順伝播
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)

add_apple_orange_layer = AddLayer()
total_price = add_apple_orange_layer.forward(apple_price, orange_price)

mul_tax_layer = MulLayer()
tax_price = mul_tax_layer.forward(total_price, tax)

# 逆伝播
d_price = 1
d_total_price, d_tax = mul_tax_layer.backward(d_price)
d_apple_price, d_orange_price = add_apple_orange_layer.backward(d_total_price)
d_apple, d_apple_num = mul_apple_layer.backward(d_apple_price)
d_orange, d_orange_num = mul_orange_layer.backward(d_orange_price)

print(tax_price)
print(d_apple_num, d_apple, d_orange, d_orange_num, d_tax)