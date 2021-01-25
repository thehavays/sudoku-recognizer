import numpy as np
import glob
import cv2
import os

import SudokuRecognizer as sr
from mnist import MNIST

K_NEIGHBOR_VALUE = 9
PCA_THRESHOLD = 70

# Set dataset path before start example  '/Home/sudoku_dataset-master' :
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SUDOKU_DATASET_DIR = ROOT_DIR + '\\sudoku_dataset'
MNIST_DATASET_DIR = ROOT_DIR + '\\mnist_dataset'

# Load data from MNIST :
mnist_data = MNIST(MNIST_DATASET_DIR)
train_images, train_labels = mnist_data.load_training()
test_images, test_labels = mnist_data.load_testing()

# Applying PCA to MNIST :
projection_matrix, transformation_matrix = sr.mnist_PCA(train_images, PCA_THRESHOLD)

dataset = []
for i in range(len(train_images)):
    dataset.append((transformation_matrix[i, :], train_labels[i]))

# Calculating confusion matrix, false positives/negatives.
confusion_matrix = sr.calculate_confusion_matrix(test_images, test_labels, dataset, projection_matrix,
                                                 K_NEIGHBOR_VALUE)

IMAGE_DIR = SUDOKU_DATASET_DIR + '\\image*.jpg'
DATA_DIR = SUDOKU_DATASET_DIR + '\\image*.dat'
ALL_IMAGE_DIRS = glob.glob(IMAGE_DIR)
ALL_DATA_DIRS = glob.glob(DATA_DIR)
len(ALL_IMAGE_DIRS)

# Accumulate accuracy for average accuracy calculation.
cumulative_acc = 0
collection_of_sudoku_arrays = []
collection_of_data = []

# Loop over all images
for img_dir, data_dir in zip(ALL_IMAGE_DIRS, ALL_DATA_DIRS):
    # Define your variables etc.:
    image_name = os.path.basename(img_dir)
    data = np.genfromtxt(data_dir, skip_header=2, dtype=int, delimiter=' ')
    img = cv2.imread(img_dir)

    # detect sudoku puzzle:
    bounding_box = sr.detect_sudoku(img)

    # Recognize digits in sudoku puzzle :
    sudoku_array = sr.recognize_sudoku(img, bounding_box, dataset, projection_matrix, K_NEIGHBOR_VALUE)

    collection_of_data.append(data)
    collection_of_sudoku_arrays.append(sudoku_array)

    # Evaluate Result for current image :
    detectionAccuracyArray = data == sudoku_array
    accPercentage = np.sum(detectionAccuracyArray) / detectionAccuracyArray.size
    cumulative_acc = cumulative_acc + accPercentage
    print(image_name + " accuracy : " + accPercentage.__str__() + "%")

# Calculate confusion matrix, false positives/negatives for Sudoku dataset here.
sr.calculate_confusion_matrix_sudoku(collection_of_sudoku_arrays, collection_of_data)

# Average accuracy over all images in the dataset :
averageAcc = cumulative_acc / len(ALL_IMAGE_DIRS)
print("dataset performance : " + averageAcc.__str__() + "%")
