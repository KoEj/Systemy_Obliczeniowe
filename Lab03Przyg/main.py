import hello
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import time

fig, ax = plot.subplots(2, 2, figsize=(10, 10))
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
originalPicture = image.imread('lenna.png')

originalPicture = originalPicture - np.min(originalPicture)
originalPicture /= np.max(originalPicture)
originalPicture *= 255
originalPicture = originalPicture.astype(int)

startTime = time.perf_counter()
pct = hello.naive_convolve(originalPicture, kernel)
finishTime = time.perf_counter()
print('Time naive convolve: ' + str(finishTime - startTime))

startTime = time.perf_counter()
pct_scnd = hello.speed_convolve(originalPicture, kernel)
finishTime = time.perf_counter()
print('Time speed convolve: ' + str(finishTime - startTime))

ax[0, 0].imshow(pct, cmap="gray")
ax[0, 1].imshow(pct_scnd, cmap="gray")

fig.savefig("Z3_0.png")
