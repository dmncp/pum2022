import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image

'''
It is possible to use a different set of photos. 
You should then remember to keep the same hierarchy of folders and their names.
If you use a different dataset, remember to run the photo_processing.py script first.
'''


def convert_vector_to_matrix(vector):
    matrix = np.reshape(vector, (photo_size, photo_size))
    matrix = np.array(matrix, dtype=np.uint8)
    img = Image.fromarray(matrix)
    img.save('./outputs/mean_img.png')


def convert_matrix_to_vector(img_path):
    img = Image.open(img_path)
    matrix = np.asarray(img)
    return matrix.flatten()


# for each image convert pixel matrix to vector and append to list
def get_pixel_vectors():
    arr = []
    for dirname in os.listdir(dataset_path):
        for filename in os.listdir(dataset_path + dirname):
            path = dataset_path + dirname + '/' + filename
            arr.append(convert_matrix_to_vector(path))
    return np.asarray(arr)


# save covariance matrix using matplotlib pyplot (png file)
def save_covariance_img(cov, title, path):
    plt.figure(figsize=(5, 5))
    plt.imshow(cov)
    plt.axis('off')
    plt.title(title)
    plt.savefig(path)
    print(f'{path} saved successfully')
    plt.clf()


# save variance using matplotlib pyplot (png file)
def save_variance_img(var, title, path):
    plt.plot(var, '.', markersize=1)
    plt.title(title)
    plt.ylabel("osiągana wartość")
    plt.xlabel("cecha")
    plt.savefig(path)
    print(f'{path} saved successfully')
    plt.clf()


# get variance values from covariance matrix (features variances are on the diagonal of the covariance matrix)
def variance(cov):
    return np.diagonal(cov)


# np.set_printoptions(threshold=sys.maxsize)
if __name__ == "__main__":
    photo_size = 28  # 28x28 px
    dataset_path = "./dataset/"
    datasets_vectors = get_pixel_vectors()  # we get a list of 60 points in 10_000 dimensional space

    pca = PCA(3)
    pca.fit_transform(datasets_vectors)  # Fit the model with X axis (centering) and transform

    # covariance matrix before and after PCA transform
    cov_before = np.cov(datasets_vectors.T)
    cov_after = pca.get_covariance()
    print(f"Covariance Before: \n {cov_before}", end='\n\n')
    print(f"Covariance After: \n {cov_after}", end='\n\n')
    save_covariance_img(cov_before, 'Macierz kowariancji przed transformacją', './outputs/cov_matrix_before.png')
    save_covariance_img(cov_after, 'Macierz kowariancji po transformacji', './outputs/cov_matrix_after.png')

    # get variance before and after PCA transform
    var_before = variance(cov_before)
    var_after = variance(cov_after)
    print(f'Variance Before: \n {var_before}', end='\n\n')
    print(f'Variance After: \n {var_after}', end='\n\n')
    save_variance_img(var_before, "Wariancja przed transformacją", './outputs/var_before.png')
    save_variance_img(var_after, "Wariancja po transformacji", './outputs/var_after.png')

    '''we can calculate the mean values of each column using:
           mean_vector = np.mean(datasets_vectors, axis=0, dtype=int)
       but we can also use PCA: pca.mean_
    '''
    # Save image from mean_vector
    convert_vector_to_matrix(pca.mean_)

    

