import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score, rand_score
from dataset_generator import get_features
import numpy as np


def print_and_save_chart(x, y, err, x_title, y_title, title, filename):
    plt.figure(figsize=(16, 8))
    plt.errorbar(x, y, yerr=err)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    print(f'Saving {filename} file...')
    plt.savefig(f'./outputs/{filename}')
    print('file saved')


def elbow_method(Kmin=1, Kmax=30, no_iterations=20):
    K = range(Kmin, Kmax)
    fowlkes_mallows_avg = []
    fowlkes_mallows_std = []
    random_std = []
    random_avg = []
    for k in K:
        print(f"ELBOW FOR K={k}:")
        fowlkes_scores = []
        random_scores = []
        for i in range(no_iterations):
            print(f"iteration nr.{i}")
            kmeans = KMeans(n_clusters=k)
            predict = kmeans.fit_predict(get_features())
            labels = kmeans.labels_
            print(predict, labels)

            fowlkes_scores.append(fowlkes_mallows_score(labels, predict))
            random_scores.append(rand_score(labels, predict))

        print(f"Fowlkes: {fowlkes_scores}\n  Std: {np.std(fowlkes_scores)}\n  Avg: {np.average(fowlkes_scores)}")
        print(f"Randoms: {random_scores}\n   Std: {np.std(random_scores)}\n   Avg: {np.average(random_scores)}")
        fowlkes_mallows_std.append(np.std(fowlkes_scores))
        print(f"fm std: {fowlkes_mallows_std}")
        fowlkes_mallows_avg.append(np.average(fowlkes_scores))
        print(f"fm avg {fowlkes_mallows_avg}")
        random_std.append(np.std(random_scores))
        print(f"rand std {random_std}")
        random_avg.append(np.average(random_scores))
        print(f"rand avg {random_avg}")

    print(f"PRINTING fm: {fowlkes_mallows_avg}\n {fowlkes_mallows_std}")
    print_and_save_chart(K, fowlkes_mallows_avg, fowlkes_mallows_std, "k", "Fowlkes-Mallows score",
                         "Elbow method (Fowlkes-Mallows)", "elbow_fm")
    print(f"PRINTING rd: {random_avg}\n {random_std}")
    print_and_save_chart(K, random_avg, random_std, "k", "Random-Score", "Elbow method (random-score)", "elbow_random")


if __name__ == '__main__':
    elbow_method()
