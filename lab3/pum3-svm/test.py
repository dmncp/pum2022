import numpy as np

if __name__ == '__main__':
    np.random.seed(0)
    X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
    y = [1] * 10 + [-1] * 10

    print(X, y)

    sample_weight_last_ten = abs(np.random.randn(len(X)))
    sample_weight_constant = np.ones(len(X))
    # and bigger weights to some outliers
    sample_weight_last_ten[15:] *= 5
    sample_weight_last_ten[9] *= 15


    print(sample_weight_constant, sample_weight_last_ten)