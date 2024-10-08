import numpy as np
import matplotlib.pyplot as plt

curve = np.array([[463, 206,], [197, 172,], [245, 237,], [484,  99,], [359, 273,], [487,  49,], [186,  97,], [181,  47,], [479, 139,], [301, 267,], [423, 251,]])

grad = np.gradient(curve)
print(grad)


plt.scatter(curve[:, 0], curve[:, 1])
plt.show()