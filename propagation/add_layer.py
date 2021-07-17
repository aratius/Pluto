class AddLayer:
  def __init__(self):
    # 特に何も行わない
    pass

  def forward(self, x, y):
    # 乗算と違い入力値を記憶しておく必要ない
    out = x + y
    return out

  def backward(self, dout):
    # そのまま下流に流す
    dx = dout * 1
    dy = dout * 1
    return dx, dy
