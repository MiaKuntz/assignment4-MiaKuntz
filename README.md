# Assignment 4 - Identifying most similar images via Histogram Feature Extraction and K-Nearest Neighbours
This assignment focuses on designing image search algorithms of the Butterflies and Moths dataset via comparison of histogram values and k-nearest neighbours. The objective is to create two Python scripts for the extraction of colour histograms and the k-nearest neighbours, which should both be readable and reproducible, along with saving distance metrics, the three most similar images from the histogram values and three most similar images from the nearest neighbour’s calculation. The purpose is to be able to find and plot possible differences in results between the two methods of finding similar images to a target image.

## Tasks
The tasks for this assignment are to:
-	Define a particular image and extract its colour histogram and features.
-	Extract colour histograms and features for the rest of the images in the data and compare these to the first image.
-	Find the three images which are most similar to the target image and save these to the ```out``` folder along with a distance metrics file.

## Repository content
The GitHub repository contains four folders, namely the ```in``` folder, where the dataset for this assignment can be stored after downloading it, the ```out``` folder, which contains the plotted images from both the histogram and k-nearest neighbour scripts, as well as a distance metrics file, the ```src``` folder, which contains the Python scripts for histogram value extraction and k-nearest neighbour feature extraction, and the ```utils``` folder, which contains helper functions for extracting features. Additionally, the repository has a ```ReadMe.md``` file, as well ```setup.sh``` and ```requirements.txt``` files.

### Repository structure
| Column | Description|
|--------|:-----------|
| ```in``` | Folder containing the butterflies and moths dataset |
| ```out``` | Folder containing the plotted target and most similar images and distance metrics file |
| ```src```  | Folder containing python scripts for k-nearest neighbours and histograms |
| ```utils``` | Folder containing utility functions for extracting features provided by the course instructor |

## Data
The data to be used in this assignment is the Butterfly & Moths Image Classification 100 species dataset. This dataset contains 12,594 images spread across 100 categories pertaining to different species of butterflies and moths.
When downloaded, the main ```archive``` repository contains three subfolders: “train”, “test”, and “valid”, along with some additional files pertaining to the authors own work with the dataset. To download the data, please follow this link:

https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species?select=EfficientNetB0-100-%28224+X+224%29-+97.59.h5

To access and prepare the data for use in the script, please; Create a user on the website, download the data, and import the data into the ```in``` folder in the repository. Please note that for the purpose of this assignment, only the “train” folder is used and kept in the repository, as this folder contains the most images and was therefore the most useful to the purpose of this assignment.

## Methods
The following is a description of parts of my code where additional explanation of my decisions on arguments and functions may be needed than what is otherwise provided in the code. 

To extract and normalize histogram for the butterflies and moths in the dataset I roughly follow the same outline of code as used in “Assignment 1”. The major difference in this self-assigned assignment is the way of the dataset, as the code needs to be able to iterate over several folders and calculate whether the most similar images are in another folder than the base folder. 

To find most similar images by using k-nearest neighbour the code first loads the VGG16 model, as well as stores the images in an empty list. It thereafter extracts features and uses the k-nearest neighbour on these features, along with calculating and saving the indices of the nearest neighbours excluding the target image. 

For both scripts plotting the three most similar images are done in a way of the plots being able to show the base folder and image number for the four displayed images, as well as having the titles of the plots show which target image was chosen for that specific example.

## Usage
### Prerequisites and packages
To be able to reproduce and run this code, make sure to have Bash and Python3 installed on whichever device it will be run on. Please be aware that the published scripts were made and run on a MacBook Pro from 2017 with the MacOS Ventura package, and that all Python code was run successfully on version 3.11.1.

The repository will need to be cloned to your device. Before running the code, please make sure that your Bash terminal is running from the repository; After, please run the following from the command line to install and update necessary packages:
bash setup.sh

### Running the script
My system requires me to type “python3” in the beginning of my commands, and the following is therefor based on this. To run the scripts from the command line please be aware of your specific system, and whether it is necessary to type “python3”, “python”, or something else in front of the commands. As the scripts uses argparse to choose the target image please be aware to include the image path after –target when running. Now run:

	python3 src/hist.py --target 

And:

    python3 src/knn.py --target 

This will active the scripts. When running, it will go through each of the functions in the order written in my main functions. That is:
-	Creating parser argument for target image.
-	Getting target image, and at some point, extract the target image number.
-	The histogram script is then:
    - Calling function for target image and then for all images.
    - Saving and plotting the top 3 most similar images with target images.
-	The k-nearest neighbour script is then:
    - Loading the model and all images.
    - Finding the index for the chosen target image.
    - Extracting features for all images and adding the top three most similar images to index.
    - Saving and plotting the top 3 most similar images with target images.

## Results
As the repository contains the results of two different target images, I will go over each example individually.

The first target image chosen was from the Mestra butterfly species, where image “001.jpg” was chosen. Both a distance metrics file and a plot of the target image and its most similar images were created when running the hist.py script. The distance metrics file doesn’t tell me much since the scores of the distance calculation are difficult to evaluate. But the file does include the similar images file paths, which tells us whether the most similar images are from the same species as the target image or not. This leads to the plotted target image and its most similar images, as this provides me with a visual of the images as well as their file paths. The results are surprising, as none of the similar images chosen by way of histogram feature extraction are of the same species as the target image. Additionally, all similar images have a black background, whereas the target image contains little, if any, black colour. Butterfly itself is yellow in colour, which shows in two of the similar images, where the species Comet Moth also has a yellowish colour to its wings. 

The difference in the plotted images generated from running the knn.py script versus the hist.py script is clearly visible. Although only one of the similar images is of the Mestra species like the target image, all the similar images in ways of colours and shapes seems nearer the target image than those found from the features extracted using the histogram method, where I would argue that both wing and background colour come close to that of the target image. 

The second target images chosen was from the Popinjay butterfly species, where image “001.jpg” was chosen. Once again both a distance metrics file and a plot of the target image and its most similar images were created when running the hist.py script, where the plotted images are “easier” to evaluate on compared to the distance scores. The most similar image is also of the Popinjay species, and the other two similar images share many of the same colours as the target image, where the Copper Tail moth was probably chosen due to the black background in the image, which fits the black wings of the Popinjay. 

Once again, the difference in result when running the k-nearest neighbour script is clearly visible when looking at the plotted most similar images. All are of the same species as the target image, and the model has been able to find another image in the dataset nearly identical to the target image, as well as having the other two similar images being near identical. 

Both methods have their own merits, but the result of this assignment clearly shows that the k-nearest neighbour method comes closest to providing the most similar images compared to the chosen target image in both examples. 

