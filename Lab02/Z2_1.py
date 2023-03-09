from functools import partial
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import multiprocessing
import time

kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
originalPicture = image.imread('lenna.png')
paddingPicture = np.pad(originalPicture, 1, mode='edge')


def convolutionWithoutPartial(coordinates):
    ky, kx = np.shape(kernel)
    y_c, x_c = coordinates
    paddingImage = np.sum(kernel * paddingPicture[y_c: ky + y_c, x_c: kx + x_c])
    return paddingImage


def convolution(coordinates, img, filterKernel):
    # PARTIAL
    ky, kx = np.shape(filterKernel)
    y_c, x_c = coordinates
    img = np.sum(filterKernel * img[y_c: ky + y_c, x_c: kx + x_c])
    return img


if __name__ == "__main__":
    fig, ax = plot.subplots(1, 2, figsize=(10, 10))
    cords = []
    donePicture = np.zeros(np.shape(originalPicture))
    result = np.zeros(np.shape(originalPicture))
    ax[0].imshow(paddingPicture, cmap="gray")

    for x_cords in range(originalPicture.shape[0]):
        for y_cords in range(originalPicture.shape[1]):
            cords.append((y_cords, x_cords))

    print('Started')
    startTime = time.perf_counter()
    with multiprocessing.Pool(processes=4) as pool:
        # PARTIAL
        # result = pool.map(partial(convolution, kernelArray=kernel, paddingImage=paddingPicture), cords)
        result = pool.map(convolutionWithoutPartial, cords)
    finishTime = time.perf_counter()
    print('Ended')
    print('Time: ' + str(finishTime - startTime))
    # print('Time: ' + str(finishTime - startTime))

    for matrixCords, (x, y) in enumerate(cords):
        donePicture[x, y] = result[matrixCords]

    ax[1].imshow(donePicture, cmap="gray")
    fig.savefig('Z2_1.png')
