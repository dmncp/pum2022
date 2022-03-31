from PIL import Image, ImageOps
import os


def resize_image(img_path):
    # Create an Image Object from an Image
    img = Image.open(img_path)

    # Make the new 100x100 image
    resized_img = img.resize((photo_size, photo_size))

    # Save the cropped image
    resized_img.save(img_path)


def convert_to_grayscale(img_path):
    # Create an Image Object from an Image
    img = Image.open(img_path)

    # convert to grayscale and save
    gray_image = ImageOps.grayscale(img)
    gray_image.save(img_path)


'''
These functions should only be run when the starting data set has not been preprocessed.
'''
if __name__ == "__main__":
    photo_size = 100  # 28x28 px
    dataset_path = "./dataset/"

    # for all sets (set1, set2, set3)
    for dirname in os.listdir(dataset_path):
        for filename in os.listdir(dataset_path + dirname):
            path = dataset_path + dirname + '/' + filename
            resize_image(path)
            convert_to_grayscale(path)
