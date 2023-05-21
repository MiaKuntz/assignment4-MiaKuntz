# importing operating system
import os
# importing sys
import sys
sys.path.append(".")
# adding path to utils folder
import utils.extract_features as ef
# importing argparse
import argparse
# data analysis
from tqdm import notebook
# tensorflow
from tensorflow.keras.applications.vgg16 import (VGG16, 
                                                 preprocess_input)
# sklearn
from sklearn.neighbors import NearestNeighbors
# matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# defining function to load model and images
def load_model_and_images():
    # loading VGG16 model
    model = VGG16(weights='imagenet', 
                include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3))
    # defining path to training data
    root_dir = os.path.join("in", "archive", "train")
    # creating list to store filenames
    filenames = []
    # iterating over all folders and files in train folder
    for dirpath, _, filenames_in_dir in sorted(os.walk(root_dir)):
        # adding all files in current folder to filenames list if they are jpg files
        filenames.extend([os.path.join(dirpath, name) for name in sorted(filenames_in_dir) if name.endswith('.jpg')])
    return model, filenames

def extract_image_number_from_path(path):
    # Extracts the image number from the file path
    filename = os.path.basename(path)
    image_number = os.path.splitext(filename)[0]
    return int(image_number)

def feature_extractor(model, filenames, target_idx):
    # creating list to store features for each image
    feature_list = []
    # iterating over all images in filenames list
    for i in notebook.tqdm(range(len(filenames)), position=0, leave=True):
        # extracting features from image
        feature_list.append(ef.extract_features(filenames[i], model))
    # using k-nearest neighbors to find similar images
    neighbors = NearestNeighbors(n_neighbors=10,
                                algorithm='brute',
                                metric='cosine').fit(feature_list)
    # calculating distances and indices of k-nearest neighbors from feature_list
    _, indices = neighbors.kneighbors([feature_list[target_idx]])
    # saving indices of nearest neighbors
    idxs = indices[0][1:4]
    return idxs

def plot_images(filenames, idxs, target_idx, target_image_number):
    basefolder = os.path.basename(os.path.dirname(filenames[target_idx]))
    # create 2x2 plot with target image and 3 most similar images
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Target Image {target_image_number:03d} ({basefolder}) and its 3 most similar images")
    # plotting target image
    axs[0, 0].imshow(mpimg.imread(filenames[target_idx]))
    axs[0, 0].set_title(f'Target Image {target_image_number:03d}: {os.path.basename(os.path.dirname(filenames[target_idx]))}/{os.path.basename(filenames[target_idx])}')
    # plotting three most similar images
    axs[0, 1].imshow(mpimg.imread(filenames[idxs[0]]))
    axs[0, 1].set_title(f'Similar Image 1: {os.path.basename(os.path.dirname(filenames[idxs[0]]))}/{os.path.basename(filenames[idxs[0]])}')
    axs[1, 0].imshow(mpimg.imread(filenames[idxs[1]]))
    axs[1, 0].set_title(f'Similar Image 2: {os.path.basename(os.path.dirname(filenames[idxs[1]]))}/{os.path.basename(filenames[idxs[1]])}')
    axs[1, 1].imshow(mpimg.imread(filenames[idxs[2]]))
    axs[1, 1].set_title(f'Similar Image 3: {os.path.basename(os.path.dirname(filenames[idxs[2]]))}/{os.path.basename(filenames[idxs[2]])}')
    # saving plot as a single image
    plt.savefig(f"out/{basefolder}_target_{target_image_number:03d}_and_knn_images.png")

def main():
    # defining argument parser
    parser = argparse.ArgumentParser(description='Find similar images to a target image using k-nearest neighbor.')
    # adding arguments
    parser.add_argument('--target', type=str, help='Path to the target image file.')
    # parsing arguments
    args = parser.parse_args()
    # target image file
    target_file = args.target
    # loading model and images
    model, filenames = load_model_and_images()
    # finding the index of the target image in the filenames list
    target_idx = filenames.index(target_file)
    # extracting the image number from the target path file
    target_image_number = extract_image_number_from_path(target_file)
    # extracting features from images
    idxs = feature_extractor(model, filenames, target_idx)
    # plotting images
    plot_images(filenames, idxs, target_idx, target_image_number)


if __name__ == "__main__":
    main()
