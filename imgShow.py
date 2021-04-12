import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread("./images/arrow.png")
plt.imshow(img)
plt.show()