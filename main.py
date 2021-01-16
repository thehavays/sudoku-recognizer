# DO NOT IMPORT ANY OTHER LIBRARY.
import numpy as np
import glob
import cv2
import os

import SudokuRecognizer as sr
from mnist import MNIST

# Define your functions here if required :


# Set dataset path before start example  '/Home/sudoku_dataset-master' :
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sudoku_dataset_dir = ROOT_DIR + '\\sudoku_dataset'
MNIST_dataset_dir = ROOT_DIR + '\\mnist_dataset'

mndata = MNIST(MNIST_dataset_dir)
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# Apply PCA to MNIST :
# use sr.mnistPCA() that you applier for transformation
# classify test set with any method you choose (hint simplest one : nearest neighbour)
# report the outcome
# Calculate confusion matrix, false postivies/negatives.
# print(reporting_results)


image_dirs = sudoku_dataset_dir + '\\image*.jpg'
data_dirs = sudoku_dataset_dir + '\\image*.dat'
IMAGE_DIRS = glob.glob(image_dirs)
DATA_DIRS = glob.glob(data_dirs)
len(IMAGE_DIRS)

# Define your variables etc. outside for loop here:

# Accumulate accuracy for average accuracy calculation.
cumulativeAcc = 0

# Loop over all images
for img_dir, data_dir in zip(IMAGE_DIRS, DATA_DIRS):
    # Define your variables etc.:
    image_name = os.path.basename(img_dir)
    data = np.genfromtxt(data_dir, skip_header=2, dtype=int, delimiter=' ')
    img = cv2.imread(img_dir)

    # detect sudoku puzzle:
    boundingBox = sr.detectSudoku(img)

    # Uncomment this section if you would like to see resulting bounding boxes.
    # cv2.rectangle(img, boundingBox[0], boundingBox[1], (0, 0, 255), 2)
    # cv2.imshow(image_name, img)
    # cv2.waitKey()

    # Recognize digits in sudoku puzzle :
    sudokuArray = sr.RecognizeSudoku(img)

    # Evaluate Result for current image :

    detectionAccuracyArray = data == sudokuArray
    accPercentage = np.sum(detectionAccuracyArray) / detectionAccuracyArray.size
    cumulativeAcc = cumulativeAcc + accPercentage
    print(image_name + " accuracy : " + accPercentage.__str__() + "%")

# Calculate confusion matrix, false postivies/negatives for Sudoku dataset here.


# Average accuracy over all images in the dataset :
averageAcc = cumulativeAcc / len(IMAGE_DIRS)
print("dataset performance : " + averageAcc.__str__() + "%")
