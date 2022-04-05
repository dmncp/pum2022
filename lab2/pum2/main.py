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


def convert_vector_to_matrix(vector, title):
    matrix = np.reshape(vector, (photo_size, photo_size))
    matrix = np.array(matrix, dtype=np.uint8)
    img = Image.fromarray(matrix)
    img.save(title)


def convert_matrix_to_vector(img_path):
    img = Image.open(img_path)
    matrix = np.asarray(img)
    return matrix.flatten()


# for each image convert pixel matrix to vector and append to list
def get_pixel_vectors():
    arr, sets = [], []
    for dirname in os.listdir(dataset_path):
        subset = []
        for filename in os.listdir(dataset_path + dirname):
            path = dataset_path + dirname + '/' + filename
            vector = convert_matrix_to_vector(path)
            arr.append(vector)
            subset.append(vector)
        sets.append(subset.copy())
        subset.clear()

    return np.asarray(arr), sets


# save covariance matrix using matplotlib pyplot (png file)
def save_covariance_img(cov, title, path):
    plt.figure(figsize=(6, 6))
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


# use PCA with no_components and fit_transform for dataset
def pca_with_transform(no_components, dataset):
    p = PCA(no_components) if no_components else PCA()
    p_fit_transf = p.fit_transform(dataset)
    return p, p_fit_transf


# save photos to png file after reduction
def save_reduced_photos(data_sets, path):
    fig, ax = plt.subplots(3, 20, figsize=(20, 4))
    for i, d_set in enumerate(data_sets):
        for j, x in enumerate(d_set):
            ax[i][j].imshow(x, 'gray')
            ax[i][j].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(path)
    plt.clf()
    print(f'{path} saved successfully')


# reduction of dimensions and save to png
def reduction(no_components):
    inverted_subset = []
    inverted = []

    for data_set in data_sets:
        for i in data_set:
            pca3, pca_fit3 = pca_with_transform(no_components, i.reshape((photo_size, photo_size)))
            inverted_subset.append(pca3.inverse_transform(pca_fit3))
        inverted.append(inverted_subset.copy())
        inverted_subset.clear()
    save_reduced_photos(inverted, f'./outputs/reduction{no_components}.png')


def visualization_2d():
    pca2, fit = pca_with_transform(2, datasets_vectors)
    set1_x, set1_y = fit[0:20, 0], fit[0:20, 1]
    set2_x, set2_y = fit[20:40, 0], fit[20:40, 1]
    set3_x, set3_y = fit[40:60, 0], fit[40:60, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(set1_y, set1_x, marker='o', color='red')
    plt.scatter(set2_y, set2_x, marker='o', color='green')
    plt.scatter(set3_y, set3_x, marker='o', color='blue')
    plt.title("czerwony:sandaly, zielony:trampki, niebieski:trapery")
    plt.savefig('./outputs/visualization2d.png')
    print('2d visualization generated successfully')


# np.set_printoptions(threshold=sys.maxsize)
if __name__ == "__main__":
    photo_size = 64  # 28x28 px
    dataset_path = "./dataset/"
    datasets_vectors, data_sets = get_pixel_vectors()  # we get a list of 60 points in photo_size**2 dimensional space

    pca, pca_fit = pca_with_transform(3, datasets_vectors)  # Fit the model with X axis (centering) and transform

    # covariance matrix before and after PCA transform
    cov_before = np.cov(datasets_vectors.T)
    cov_after = pca.get_covariance()
    # print(f"Covariance Before: \n {cov_before}", end='\n\n')
    # print(f"Covariance After: \n {cov_after}", end='\n\n')
    save_covariance_img(cov_before, 'Macierz kowariancji przed transformacją', './outputs/cov_matrix_before.png')
    save_covariance_img(cov_after, 'Macierz kowariancji po transformacji', './outputs/cov_matrix_after.png')

    # get variance before and after PCA transform
    var_before = variance(cov_before)
    var_after = variance(cov_after)
    # print(f'Variance Before: \n {var_before}', end='\n\n')
    # print(f'Variance After: \n {var_after}', end='\n\n')
    save_variance_img(var_before, "Wariancja przed transformacją", './outputs/var_before.png')
    save_variance_img(var_after, "Wariancja po transformacji", './outputs/var_after.png')

    # Save image from mean_vector
    convert_vector_to_matrix(pca.mean_, './outputs/mean_img.png')

    # principal components
    principal_components = pca.components_  # that is the same
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(8, 10))
    for idx, pc in enumerate(principal_components):
        axes[idx % 3][idx // 3].imshow(pc.reshape(photo_size, photo_size), cmap="gray")
        axes[idx % 3][idx // 3].axis('off')
    plt.savefig("./outputs/pc.png")

    # reduction of dimensions
    reduction(3)
    reduction(9)
    reduction(27)

    # projecting a set onto a 2D plane
    visualization_2d()
