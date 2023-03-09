import z3_0
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
convPicture = z3_0.naive_convolve(originalPicture, kernel)
finishTime = time.perf_counter()

time_naive = '\nTime naive: ' + str(finishTime - startTime)
print(time_naive)

with open('times.txt', 'a') as f:
    f.writelines(time_naive)

ax[1].imshow(convPicture, cmap="gray")
fig.savefig("Z3_0.png")
