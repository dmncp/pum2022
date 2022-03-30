import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
# import cv2
from sklearn.decomposition import PCA

from keras.datasets import fashion_mnist
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()


class Data():
    shirts = []
    t_shirts = []
    coats = []

    def __init__(self):
        shirts = self.shirts
        t_shirts = self.t_shirts
        coats = self.coats
        while len(shirts) < 20 or len(t_shirts) < 20 or len(coats) < 20:
            i = np.random.randint(len(testy))
            if testy[i] == 6 and len(shirts) < 20:
                shirts.append(testX[i])
            elif testy[i] == 0 and len(t_shirts) < 20:
                t_shirts.append(testX[i])
            elif testy[i] == 4 and len(coats) < 20:
                coats.append(testX[i])


def show_variance(var):
    plt.plot(var, '.', markersize=1)
    plt.title('Wariancja')
    plt.ylabel('wartość')
    plt.xlabel('cecha')
    plt.show()


def show_cov(cov):
    plt.figure(figsize=(6,6))
    plt.imshow(cov, 'gray')
    plt.axis('off')
    plt.title('Macierz kowariancji')
    plt.show()


if __name__ == '__main__':
    data = Data()
    fig, ax = plt.subplots(3, 20, figsize=(20, 4))
    for i in range(20):
        ax[0][i].imshow(data.shirts[i], 'gray')
        ax[0][i].axis('off')

        ax[1][i].imshow(data.t_shirts[i], 'gray')
        ax[1][i].axis('off')

        ax[2][i].imshow(data.coats[i], 'gray')
        ax[2][i].axis('off')

    # plt.subplots_adjust(wspace=0, hspace=0)

    X = []
    for i in range(20):
        X.append(np.asarray(data.shirts[i]).flatten())
        X.append(np.asarray(data.t_shirts[i]).flatten())
        X.append(np.asarray(data.coats[i]).flatten())
    X = np.asarray(X)

    pca1 = PCA()
    pca1.fit(X)
    pca2 = PCA(n_components=3)
    pca2.fit_transform(X)

    cov_before = np.cov(X.T)
    print(cov_before)

    cov_after = pca2.get_covariance()
    print(cov_after)

    print('Macierz kowariancji przed transformacją:')
    show_cov(cov_before)
    print('min =',np.min(cov_before))
    print('max =', np.max(cov_before))
    print('mean =', np.mean(cov_before))
    print('std =', np.std(cov_before))


    print('Macierz kowariancji po transformacji:')
    show_cov(cov_after)
    print('min =',np.min(cov_after))
    print('max =', np.max(cov_after))
    print('mean =', np.mean(cov_after))
    print('std =', np.std(cov_after))

    variance_before = []
    for i in range(784):
        variance_before.append(cov_before[i,i])
    print('Wariancja przed transformacją:')
    show_variance(variance_before)

    variance_after = []
    for i in range(784):
        variance_after.append(cov_after[i,i])
    print('Wariancja po transformacji:')
    show_variance(variance_after)