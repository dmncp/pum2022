import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score, rand_score
from dataset_generator import get_features
import numpy as np


def print_and_save_chart(x, K, x_title, y_title, title, filename):
    plt.figure(figsize=(16, 8))
    plt.plot(K, x, 'bx-')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.savefig(f'./outputs/{filename}')


def elbow_method(Kmin=1, Kmax=30, no_iterations=100):
    K = range(Kmin, Kmax)
    distortions_avg = []
    distortions_std = []
    predict_avg = []
    predict_std = []

    for k in K:
        print(f"ELBOW FOR K={k}:")
        results = []
        predict_results = []
        for i in range(no_iterations):
            print(f"iteration nr.{i}")
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(get_features())
            results.append(kmeans.inertia_)

            # compare with predict result
            predict_results.append(kmeans.predict(get_features()))

        print(results)
        print(np.std(results))
        print(np.average(results))
        print(predict_results)
        print(np.std(predict_results))
        print(np.average(predict_results))
        distortions_avg.append(np.average(results))
        distortions_std.append(np.std(results))
        predict_avg.append(np.average(predict_results))
        predict_std.append(np.std(predict_results))

    print_and_save_chart(distortions_avg, K, "k", "Distortion AVG", "Elbow method AVG", "elbow_avg")
    print_and_save_chart(distortions_std, K, "k", "Distortion STD", "Elbow method STD", "elbow_std")
    print_and_save_chart(predict_avg, K, "k", "Distortion AVG (predict)", "Elbow method AVG predict", "elbow_avg_predict")
    print_and_save_chart(predict_std, K, "k", "Distortion STD (predict)", "Elbow method STD predict", "elbow_std_predict")

    print(f"Fowlkes-Mallows score: {fowlkes_mallows_score(distortions_avg, predict_avg)}")
    print(f"Rand score: {rand_score(distortions_avg, predict_avg)}")


if __name__ == '__main__':
    elbow_method()
