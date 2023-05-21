# Author: Mia Kuntz
# Date hand-in: 24/5 - 2023

# Description: This script finds the three most similar images to a target image using k-nearest neighbor.
# The script is run from the command line and takes one argument: the path to the target image file.
# The script outputs a plot with the target image and its three most similar images, as well as a csv file with the distance metric for all images.

# importing operating system
import os
# importing argparse
import argparse
# importing cv2
import cv2
# importing pandas
import pandas as pd
# importing matplotlib
import matplotlib.pyplot as plt

# defining function for target image
def target_image(target_img):
    # reading target image
    target = cv2.imread(target_img)
    # calculating histogram of target image
    hist_target = cv2.calcHist([target], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # normalizing histogram of target image
    hist_target_norm = cv2.normalize(hist_target, hist_target, 0, 1.0, cv2.NORM_MINMAX)
    return hist_target_norm

# defining function for all images
def all_images(hist_target_norm):
    # defining root directory
    root_dir = os.path.join("in", "archive", "train")
    # creating empty list for histogram value
    hist_value = []
    # creating for loop for directory path, directory name and file name in directory
    for dirpath, _, filenames_in_dir in sorted(os.walk(root_dir)):
        # specifying for loop for file name in file name in directory
        for filename in filenames_in_dir:
            # creating if statement for file name endswith .jpg
            if filename.endswith(".jpg"):
                # getting relative folder path
                folder = os.path.relpath(dirpath, root_dir)
                # getting file path
                filepath = os.path.join(folder, filename)
                # getting complete file path
                butterfly_jpg = cv2.imread(os.path.join(root_dir, filepath))
                # calculating histogram of butterfly images
                hist_butterflies = cv2.calcHist([butterfly_jpg], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                # normalizing histogram of butterfly images
                hist_butterflies_norm = cv2.normalize(hist_butterflies, hist_butterflies, 0, 1.0, cv2.NORM_MINMAX)
                # calculating distance between target image and butterfly images
                comp_value = round(cv2.compareHist(hist_target_norm, hist_butterflies_norm, cv2.HISTCMP_CHISQR), 2)
                # appending histogram value
                hist_value.append((comp_value, folder, filename))
    # sorting histogram values
    hist_value = sorted(hist_value, key=lambda x: x[0])[:4]
    # creating dataframe for histogram values
    df = pd.DataFrame(hist_value, columns=["Distance", "Folder", "Filename"])
    return root_dir, df

# defining function for plotting images
def plot_images(images, filenames, output_path, target_image_number, target_folder, df):
    # creating figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # setting title
    fig.suptitle(f"Target Image {target_image_number} ({target_folder}) and its 3 most similar images")
    # creating for loop for images and axes
    for i, ax in enumerate(axs.flatten()):
        # creating if statement for i < len(images)
        if i < len(images):
            # showing images
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            # creating if statement for i == 0
            if i == 0:
                # setting title
                ax.set_title(f"Target Image {target_image_number}: {filenames[i]}")
            # creating else statement
            else:
                # setting title
                ax.set_title(f"Similar Image {i}: {filenames[i]}")
        # creating else statement
        else:
            # hiding axes
            ax.axis('off')
    # saving figure
    fig.savefig(output_path)
    # saving dataframe as csv file
    df.to_csv(f"out/{target_folder}_target_{target_image_number}_distance_metric.csv", index=False)

# defining function for saving top 3 images
def save_top3(df, target_image_path, root_dir, target_image_number):
    # creating empty list for images and filenames
    images = []
    filenames = []
    # reading target image
    target_image = cv2.imread(target_image_path)
    # appending target image and target image path to images and filenames
    images.append(target_image)
    target_folder = os.path.basename(os.path.dirname(target_image_path))
    # appending target image path to filenames
    filenames.append(os.path.join(target_folder, os.path.basename(target_image_path)))
    # creating dataframe for top 3 images
    top3_df = df[df['Filename'] != os.path.basename(target_image_path)].head(3)
    # creating for loop for row in top3_df.iterrows()
    for _, row in top3_df.iterrows():
        # getting folder name
        folder = os.path.basename(row['Folder'])
        # getting filename
        filename = row['Filename']
        # getting image path
        img_path = os.path.join(root_dir, row['Folder'], filename)
        # reading image
        img = cv2.imread(img_path)
        # creating if statement for img is not None and img.shape[0] > 0 and img.shape[1] > 0
        if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
            # resizing image
            img = cv2.resize(img, (target_image.shape[1], target_image.shape[0]))
            # appending image to images
            images.append(img)
            # appending image path to filenames
            filenames.append(os.path.join(folder, filename))
    # creating output path for images and filenames with target image number and target folder
    output_path = os.path.join("out", f"{target_folder}_target_{target_image_number}_and_hist_images.png")
    # calling function for plotting images
    plot_images(images, filenames, output_path, target_image_number, target_folder, df)

# defining main function
def main():
    # defining argument parser
    parser = argparse.ArgumentParser(description='Find similar images to a target image using histogram comparison.')
    parser.add_argument('--target', type=str, help='Path to the target image file.')
    args = parser.parse_args()
    # calling function for target image
    target_img = args.target
    # getting target image number
    target_image_number = os.path.splitext(os.path.basename(target_img))[0]
    # calling function for target image
    hist_target_norm = target_image(target_img)
    # calling function for all images
    root_dir, df = all_images(hist_target_norm)
    # calling function for saving top 3 images
    save_top3(df, target_img, root_dir, target_image_number)

if __name__ == "__main__":
    main()

# Command line arguments example: 
# python3 src/hist.py --target in/archive/train/JULIA/001.jpg