def numerical_diff(f, x):
    # プログラムで認識できる程度の小さな値
    h = 1e-4  #1*10^-4

    # 誤差を小さくするため中心差分を取る
    return (f(x+h) - f(x - h)) / (2 * h)

def fn (x):
    return x*x

# 偏微分
def fn_tmp(x):
    return x[0]**2 + x[1]**2

d = numerical_diff(fn, 50)
print(d)
