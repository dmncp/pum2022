
# Principal Component Analysis
from numpy import array
from sklearn.decomposition import PCA



if __name__ == '__main__':
    # define a matrix
    A = array([[1, 2], [3, 4], [5, 6]])
    # create the PCA instance
    pca = PCA(2)
    # fit on data
    pca.fit(A)
    # access values and vectors
    print(pca.components_, end='\n\n')  # == print(vectors)
    print(pca.explained_variance_, end='\n\n')  # == print(values)
    # transform data
    B = pca.transform(A)
    print(pca.get_covariance(), end='\n\n')  # print(V)
    print(B, end='\n\n')  # == print(P.T)
    print(pca.mean_, end='\n\n')  # == print(M)

    print("+++++++++++++++++++++")

from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# calculate the mean of each column
M = mean(A.T, axis=1)
print(M)
# center columns by subtracting column means
C = A - M
print(C)
# calculate covariance matrix of centered matrix
V = cov(C.T)
print(V)
# eigendecomposition of covariance matrix
values, vectors = eig(V)
print(vectors)
print(values)
# project data
P = vectors.T.dot(C.T)
print(P.T)

