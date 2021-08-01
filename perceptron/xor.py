from _and import AND
from _or import OR
from _nand import NAND

def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y



print(XOR(0, 0))  # 0
print(XOR(1, 0))  # 1
print(XOR(0, 1))  # 1
print(XOR(1, 1))  # 0