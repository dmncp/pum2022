import os
import sys

from sklearn.decomposition import PCA

import numpy as np
from PIL import Image

'''
It is possible to use a different set of photos. 
You should then remember to keep the same hierarchy of folders and their names.
If you use a different dataset, remember to run the photo_processing.py script first.
'''


def convert_vector_to_matrix(vector):
    matrix = np.reshape(vector, (100, 100))
    matrix = np.array(matrix, dtype=np.uint8)
    img = Image.fromarray(matrix)
    img.save('./mean_img.png')


def convert_matrix_to_vector(img_path):
    img = Image.open(img_path)
    matrix = np.asarray(img)
    return matrix.flatten()


# for each image convert pixel matrix to vector and append to list
def get_pixel_vectors():
    for dirname in os.listdir(dataset_path):
        for filename in os.listdir(dataset_path + dirname):
            path = dataset_path + dirname + '/' + filename
            datasets_vectors.append(convert_matrix_to_vector(path))


np.set_printoptions(threshold=sys.maxsize)
if __name__ == "__main__":
    dataset_path = "./dataset/"
    datasets_vectors = []

    get_pixel_vectors()  # we get a list of 60 points in 10_000 dimensional space

    pca = PCA(60)  # why 60? and why not 10_000?
    pca.fit(datasets_vectors)

    '''we can calculate the mean values of each column using:
           mean_vector = np.mean(datasets_vectors, axis=0, dtype=int)
       but we can also use PCA: pca.mean_
    '''

    # Next, we need to center the values in each column by subtracting the mean column value.
    # centered = np.subtract(datasets_vectors, pca.mean_)  # todo: pytanie czy to wgl jest dobrze xD

    # Save image from mean_vector
    convert_vector_to_matrix(pca.mean_)

    

