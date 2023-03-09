import z3_1
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import time

fig, ax = plot.subplots(1, 2, figsize=(10, 10))
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
originalPicture = image.imread('lenna.png')
ax[0].imshow(originalPicture, cmap="gray")

originalPicture = originalPicture - np.min(originalPicture)
originalPicture = ((originalPicture * 255) / np.max(originalPicture)).astype(int)

startTime = time.perf_counter()
convPicture = z3_1.prange_convolve(originalPicture, kernel)
finishTime = time.perf_counter()
time_prange = '\nTime prange: ' + str(finishTime - startTime)
print(time_prange)

with open('times.txt', 'a') as f:
    f.writelines(time_prange)

ax[1].imshow(convPicture, cmap="gray")
fig.savefig("Z3_1.png")
