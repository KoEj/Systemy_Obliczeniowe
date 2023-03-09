import sklearn
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.metrics import DistanceMetric
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=1234)

colors = ListedColormap(["crimson", "mediumblue"])
color=['blue','green','cyan']

# fig, ax = plt.subplots(figsize=(8, 6))
# plt.scatter(x=X[:,0], y=X[:,1], s=100, c=y, cmap = colors)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()

cluster = np.zeros(X.shape[0])

centroids = np.random.choice(len(X), size=2, replace=False)
print(centroids)
# print(centroids)
centroidsArray = X[centroids, :]
# print(centroidsArray)
Loop = True

# plt.scatter(centroidsArray[:, 0], centroidsArray[:, 1], c='r', s = 80)
# plt.show()

while Loop is True:
    for i, x in enumerate(X):
        minDist = 9999999.99
        for i_centroid, centroid in enumerate(centroidsArray):
            dist = np.amax(DistanceMetric.get_metric("euclidean").pairwise([x[:2], centroid[:2]]))
            # dist = np.array(DistanceMetric.get_metric("euclidean").pairwise([x[:2], centroid[:2]])).max()
            # print(dist)

            if minDist > dist:
                minDist = dist
                cluster[i] = i_centroid

        newCentroidsArray = pd.DataFrame(X).groupby(by=cluster).mean().values

        if np.count_nonzero(centroidsArray - newCentroidsArray) == 0:
            Loop = False
        else:
            centroidsArray = newCentroidsArray

for c in range(2):
    plt.scatter(x=X[cluster==c, 0], y=X[cluster==c, 1], c=color[c])
plt.scatter(centroidsArray[:, 0], centroidsArray[:, 1], c='r', s = 80)
plt.show()

# mpiexec -n 4 python3 main.py