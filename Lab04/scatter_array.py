import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import seaborn as sns
from sklearn.metrics import DistanceMetric

X, y_true = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, random_state=1)


def kmeans(X, k):

    diff = 1
    cluster = np.zeros(X.shape[0])

    # select k random centroids
    np.random.seed(1)
    random_indices = np.random.choice(len(X), size=k, replace=False)
    centroids = X[random_indices, :]

    while diff:

    # for each observation
        for i, row in enumerate(X):

            mn_dist = float('inf')
            # dist of the point from all centroids
            for idx, centroid in enumerate(centroids):
                # d = np.sqrt((centroid[0]-row[0])**2 + (centroid[1]-row[1])**2)
                d = np.array(DistanceMetric.get_metric("euclidean").pairwise([row[:2], centroid[:2]])).max()
                # store closest centroid
                # print(d)
                # print(d1)
                if mn_dist > d:
                    mn_dist = d
                    cluster[i] = idx

        new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values

        # if centroids are same then leave
        if np.count_nonzero(centroids-new_centroids) == 0:
            diff = 0
        else:
            centroids = new_centroids
    return centroids, cluster

k = 4
centroids, cluster = kmeans(X, k)

# plt.scatter(X[:,0], X[:, 1])
# plt.scatter(centroids[:,0], centroids[:, 1], s=100, color='y')
sns.scatterplot(x=X[:,0], y=X[:, 1], hue=cluster, palette=["green", "red", "blue", "black"])
sns.scatterplot(x=centroids[:,0], y=centroids[:, 1], s=100, color='y')
plt.show()















# from mpi4py import MPI
# import numpy as np
#
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# nprocs = comm.Get_size()
#
# if rank == 0:
#     data = np.arange(15.0)
#
#     # determine the size of each sub-task
#     ave, res = divmod(data.size, nprocs)
#     counts = [ave + 1 if p < res else ave for p in range(nprocs)]
#
#     # determine the starting and ending indices of each sub-task
#     starts = [sum(counts[:p]) for p in range(nprocs)]
#     ends = [sum(counts[:p+1]) for p in range(nprocs)]
#
#     # converts data into a list of arrays
#     data = [data[starts[p]:ends[p]] for p in range(nprocs)]
# else:
#     data = None
#
# data = comm.scatter(data, root=0)
#
# print('Process {} has data:'.format(rank), data)