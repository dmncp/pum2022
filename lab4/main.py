import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from dataset_generator import get_features
import numpy as np


def elbow_method(Kmin=1, Kmax=3, no_iterations=10):
    K = range(Kmin, Kmax)
    distortions_avg = []
    distortions_std = []

    for k in K:
        print(f"ELBOW FOR K={k}:")
        results = []
        for i in range(no_iterations):
            print(f"iteration nr.{i}")
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(get_features())
            results.append(kmeans.inertia_)
        print(results)
        print(np.std(results))
        print(np.average(results))


if __name__ == '__main__':
    elbow_method()
