import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import time

fig, ax = plot.subplots(1, 2, figsize=(10, 10))
originalPicture = image.imread('lenna.png')
# print(originalPicture.shape)

kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
donePicture = np.zeros(originalPicture.shape)

paddingPicture = np.pad(originalPicture, 1, mode='edge')
# print(paddingPicture.shape)
ax[0].imshow(paddingPicture, cmap="gray")

pY, pX = np.shape(paddingPicture)
ky, kx = np.shape(kernel)

print('Started')
startTime = time.perf_counter()
for x in range(pX - (kx + 1)):
    for y in range(pY - (ky + 1)):
        donePicture[x, y] = np.sum(kernel * paddingPicture[x: kx + x, y: ky + y])
finishTime = time.perf_counter()
print('Ended')
print('Time: ' + str(finishTime - startTime))
# print(donePicture.shape)

# print(donePicture)


ax[1].imshow(donePicture, cmap="gray")
fig.savefig("Z2_0.png")