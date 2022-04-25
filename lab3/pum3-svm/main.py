from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset_preparing import get_dataset
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_function(classifier, title):
    h = 0.01

    X = StandardScaler().fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#000000", "#800000", "#A52A2A", "#DC143C", "#FF7F50", "#191970",
                                "#000080", "#0000CD", "#00008B", "#0000FF"])

    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(classifier, "decision_function"):
        Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=0.8, levels=200)

    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")

    # Plot the testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.6)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.text(
        xx.max() - 0.3,
        yy.min() + 0.3,
        ("%.2f" % score).lstrip("0"),
        size=15,
        horizontalalignment="right",
    )
    plt.savefig(f"./outputs/{title}.png")
    plt.clf()


def plot_support_vectors(vectors, filename):
    x1, y1 = vectors[:, 0], vectors[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x1, y1)
    # plt.scatter(x2, y2, marker="o", color="red")
    plt.savefig(f"./outputs/{filename}.png")
    plt.clf()


def get_support_vectors(clf, filename):
    support_indices = clf.support_
    vectors = []
    for i in support_indices:
        vectors.append(x[i])
    plot_support_vectors(np.array(vectors), filename)


def linear_svm(C_range):
    # with a strong emphasis on the samples on the "right" side
    # "C" is the penalty parameter - smaller C creates small margin and larger C creates larger margin
    C_value = 0.01
    while C_value <= C_range:
        linear = svm.SVC(kernel="linear", C=C_value)
        plot_decision_function(linear, f"C={C_value}")
        C_value *= 10


# svm custom kernel - the product of the first component of both vectors, omitting the second component
def custom_kernel():
    def kernel_fun(X, Y):  # todo: o to chodziło??
        first_X = np.array([[i[0] for i in X]]).T
        first_Y = np.array([[i[0] for i in Y]])

        return np.dot(first_X, first_Y)

    custom = svm.SVC(kernel=kernel_fun)
    plot_decision_function(custom, "custom kernel")


# classic RBF (Radial Basis Function) kernel
def rbf_kernel_classic():
    # we can use "rbf" kernel from library which uses the gamma parameter instead of the sigma from the given formula
    # gamma = 1/(2*sigma^2)
    # so, if we want to use sigma = 0.5, 0.7, 1, 2 we have to use gamma = 1/8, 1/2, 1, 2 instead
    gamma_values = [0.125, 0.5, 1, 2]
    for g in gamma_values:
        gamma = svm.SVC(kernel='rbf', gamma=g)
        plot_decision_function(gamma, f"Gamma={g}")


def rbf_kernel_angular():
    def kernel_fun(X, Y):
        distance = np.sum((Y - X[:, np.newaxis]) ** 2, axis=-1)
        R_matrix = np.zeros(shape=distance.shape)
        R_matrix.fill(R)
        comparing_matrix = np.less(distance, R_matrix)

        return comparing_matrix * 1

    R_values = [0.5, 1, 1.5, 2, 2.5, 5]
    for R in R_values:
        angular = svm.SVC(kernel=kernel_fun)
        plot_decision_function(angular, f"R={R}")


def rbf_kernel_linear():
    def kernel_fun(X, Y):
        distance = np.sum((Y - X[:, np.newaxis]) ** 2, axis=-1)
        R_matrix = np.zeros(shape=distance.shape)
        R_matrix.fill(R)
        comparing_matrix = np.subtract(R, distance)
        comparing_matrix = comparing_matrix / R

        is_less = np.less(distance, R_matrix) * 1

        return np.multiply(comparing_matrix, is_less)

    R_values = [0.5, 1, 1.5, 2, 2.5, 5]
    for R in R_values:
        rbf_linear = svm.SVC(kernel=kernel_fun)
        plot_decision_function(rbf_linear, f"linearR={R}")


def rbf_kernel_angle():
    def kernel_fun(X, Y):
        x_size = X.shape[0]
        y_size = Y.shape[0]
        cosines = np.zeros(shape=(x_size, y_size))

        print(f"counting for ({x_size}, {y_size})")
        for i in range(x_size):
            for j in range(y_size):
                cosines[i][j] = np.dot(X[i], Y[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(Y[j]))

        return cosines

    rbf_angle = svm.SVC(kernel=kernel_fun)
    plot_decision_function(rbf_angle, f"angleRBF")


if __name__ == "__main__":
    dataset = get_dataset()
    dataset.save("starting_dataset")

    # step 1: find decision boundary
    x = np.array(dataset.set1 + dataset.set2)
    y = [1] * len(dataset.set1) + [-1] * len(dataset.set2)
    print(f"start X = {x.shape}, start y = {np.array(y).shape}")

    svc = svm.SVC()
    svc_linear = svm.SVC(kernel="linear")

    print("decision boundary for classic svm...")
    plot_decision_function(svc, "Decision_boundary")
    print("Done. Linear svm...")
    plot_decision_function(svc_linear, "Decision_boundary_linear")

    # get support vectors
    print("Done. Support vectors for classic version...")
    get_support_vectors(svc, "sv.png")
    print("Done. Support vectors for linear version...")
    get_support_vectors(svc_linear, "sv_linear.png")

    # margin width between classes
    margin = 1 / np.sqrt(np.sum(svc_linear.coef_ ** 2))
    print(f"Margin: {margin}")

    # step 2
    # linear svm version
    print("Linear versions for range 0.01 - 10000...")
    linear_svm(10_000)

    # custom kernel version
    print("Custom kernel...")
    custom_kernel()

    # RBF kernel
    print("RBF kernel...")
    rbf_kernel_classic()

    # angular RBF
    print("RBF kernel angular...")
    rbf_kernel_angular()

    # linear RBF function
    print("Linear RBF...")
    rbf_kernel_linear()

    # RBF angle
    print("RBF angle")
    rbf_kernel_angle()
