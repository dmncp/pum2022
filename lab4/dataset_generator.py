import umap.umap_ as umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns


def get_features():
    return embedding

print('preparing mnist')
# sns.set(context="paper", style="white")

mnist = fetch_openml("mnist_784", version=1)
print('done, now reducer')
reducer = umap.UMAP(random_state=42)
print('done, embedding')
embedding = reducer.fit_transform(mnist.data)
print('embedding done, go to main')
# fig, ax = plt.subplots(figsize=(12, 10))
# color = mnist.target.astype(int)
# plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
# plt.setp(ax, xticks=[], yticks=[])
# plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
#
# plt.show()
