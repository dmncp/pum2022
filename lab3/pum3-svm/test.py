import numpy as np

if __name__ == '__main__':
    x = np.array([[1, 4, 8, 7]])
    y = np.array([[0, 2, 3, 3]]).T

    print(np.multiply(x, y))
    print(x * y)


