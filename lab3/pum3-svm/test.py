import numpy as np

if __name__ == '__main__':
    x = np.array([[3, 4], [0, 2], [8, 7], [1, 5]], dtype='float32')
    y = np.array([[0, 2], [1, 1], [4, 5]], dtype='float32')

    x_size = x.shape[0]
    y_size = y.shape[0]

    cosines = np.zeros(shape=(x_size, y_size))

    for i in range(x_size):
        for j in range(y_size):
            cosines[i][j] = np.dot(x[i], y[j]) / (np.linalg.norm(x[i]) * np.linalg.norm(y[j]))

    print(cosines)

