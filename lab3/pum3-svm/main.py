from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

from dataset_preparing import get_dataset
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_function(classifier, title):
    h = 0.01

    X_train, X_test, y_train, y_test = train_test_split(x, y)

    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
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
    plt.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

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
    set1, set2 = [], []
    for v in vectors:
        if v in np.array(dataset.set1):
            set1.append(v)
        elif v in np.array(dataset.set2):
            set2.append(v)
    x1, y1 = np.array(set1)[:, 0], np.array(set1)[:, 1]
    x2, y2 = np.array(set2)[:, 0], np.array(set2)[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x1, y1, marker="x", color="blue")
    plt.scatter(x2, y2, marker="o", color="red")
    plt.savefig(f"./outputs/{filename}.png")
    plt.clf()


def get_support_vectors(clf, filename):
    support_vectors = clf.support_vectors_
    plot_support_vectors(support_vectors, filename)


def linear_svm():
    # with a strong emphasis on the samples on the "right" side
    linear = svm.SVC(kernel="linear", gamma=0.1)


if __name__ == "__main__":
    dataset = get_dataset()
    dataset.save("starting_dataset")

    # step 1: find decision boundary
    x = np.array(dataset.set1 + dataset.set2)
    y = [1] * len(dataset.set1) + [-1] * len(dataset.set2)

    svc = svm.SVC()
    svc_linear = svm.SVC(kernel="linear")

    plot_decision_function(svc, "Decision_boundary")
    plot_decision_function(svc_linear, "Decision_boundary_linear")

    # get support vectors
    get_support_vectors(svc, "sv.png")
    get_support_vectors(svc_linear, "sv_linear.png")

    # margin width between classes
    margin = 1 / np.sqrt(np.sum(svc_linear.coef_ ** 2))
    print(f"Margin: {margin}")

    # step 2
    # linear svm version

