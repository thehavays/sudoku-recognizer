# Eray HAVAYLAR S018954 Department of Computer Science
import cv2
import numpy as np


def calculate_mean(x):
    return np.mean(x)


# Matrix multiplication
def matrix_multiplication(a, b):
    return np.matmul(a, b)


# Pixel-wise difference between the images
def euclidean_distance(img1, img2):
    return sum((img1 - img2) ** 2)


def predict_value_knn(dataset, image_to_be_predicted, k):
    # Compute distances to every point in the data set.
    distances = [euclidean_distance(x[0], image_to_be_predicted) for x in dataset]

    # Make a list of (distance, label) tuples
    distance_label = [(np.abs(distances[i]), dataset[i][1]) for i in range(len(distances))]

    # Sort the (distance, label) tuples from low to high
    distance_label.sort(key=lambda x: x[0], reverse=False)

    for i in distance_label[0:9]:
        if np.isnan(i[0]):
            return 0

    # for i in range(10):
    #     print(distance_label[i])

    # Vote for every number and increment voted number in the candidate array
    candidates = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(candidates)):
        candidates[distance_label[i][1] - 1] += 1

    # Choose most voted array index as predicted number
    most_voted = candidates.index(max(candidates)) + 1
    return most_voted


def confusion_matrix_printer(confusion_matrix):
    recall = None
    precision = None

    print("Here is the confusion matrix :")
    print(confusion_matrix)

    # Calculating true positives, false positives, true negatives, and false negatives for every number
    #  and calculating recall and precision values
    # Calculate overall accuracy at the end
    sum_of_true_positives = 0

    for current_number in range(10):

        true_positives = 0
        false_positives = 0

        true_negatives = 0
        false_negatives = 0
        # False negatives are never used, I never say a number is not a specific number
        # I only predict number, so I say a number is probably a specific number

        for index in range(10):
            if current_number == index:
                true_positives = confusion_matrix[current_number, index]
                sum_of_true_positives += true_positives
            else:
                false_positives += confusion_matrix[current_number, index]
                true_negatives += confusion_matrix[index, current_number]

        if true_positives + false_positives != 0:
            recall = (true_positives * 100) / (true_positives + false_positives)

        if true_positives + true_negatives != 0:
            precision = (true_positives * 100) / (true_negatives + true_positives)

        print("For number ", current_number)
        print("true_positives  : ", true_positives)
        print("true_negatives  : ", true_negatives)
        print("false_positives : ", false_positives)
        print("false_negatives : ", false_negatives)
        print("recall          : ", recall)
        print("precision       : ", precision)

    return sum_of_true_positives


def calculate_confusion_matrix(test_images, test_labels, dataset, projection_matrix, k_neighbor_value):
    # Prepare the confusion matrix
    confusion_matrix = np.zeros(shape=(10, 10))
    true_counter = 0

    for i in range(len(test_images)):

        test_labels = np.array(test_labels)
        test_images = np.array(test_images)

        actual_value = test_labels[i]
        predicted_value = predict_value_knn(dataset, test_images[i].dot(projection_matrix), k_neighbor_value)

        confusion_matrix[actual_value][predicted_value] += 1

        if actual_value == predicted_value:
            true_counter += 1

        if i % 100 == 0:
            print("Finished ", i + 1, " image in total of ", len(test_images), "Summary : True counter : ",
                  true_counter)

    sum_of_true_positives = confusion_matrix_printer(confusion_matrix)

    accuracy = (sum_of_true_positives * 100) / len(test_images)

    print("overall accuracy : ", accuracy)


def calculate_confusion_matrix_sudoku(collection_of_sudoku_array, collection_of_data):
    # Prepare the confusion matrix
    confusion_matrix = np.zeros(shape=(10, 10))

    for i in range(len(collection_of_sudoku_array)):
        for j in range(9):
            for k in range(9):
                actual_value = collection_of_data[i][j][k]
                predicted_value = collection_of_sudoku_array[i][j][k]
                confusion_matrix[actual_value][predicted_value] += 1

    sum_of_true_positives = confusion_matrix_printer(confusion_matrix)

    accuracy = sum_of_true_positives / len(collection_of_sudoku_array)

    print("overall accuracy : ", accuracy)


# PCA on MNIST dataset, this function should return PCA transformation for mnist dataset.
# You can freely modify the signature of the function.
def mnist_PCA(images, threshold):
    number_of_samples = len(images)  # total sample image
    number_of_features = len(images[0])  # total pixel count

    mean_vector = np.apply_along_axis(calculate_mean, axis=0, arr=images)
    mean_matrix = np.array([mean_vector] * number_of_samples)
    deviation_matrix = images - mean_matrix

    multiply_with_transpose = np.matmul(deviation_matrix.transpose(), deviation_matrix)

    covariance_matrix = multiply_with_transpose / number_of_samples

    eig_values, eig_vectors = np.linalg.eig(covariance_matrix)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:, i]) for i in range(len(eig_values))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Calculation of explained variance
    total = sum(eig_values)
    var_exp = [(i / total) * 100 for i in sorted(eig_values, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    # This index show from which index value, we are above the threshold.
    # It means that I can represent my data approximately <threshold> percent of currency by reducing
    # the dimension of feature from 'number_of_features' to 'index'
    index = cum_var_exp.searchsorted(threshold)

    # projection matrix
    projection_matrix = eig_pairs[0][1].reshape(number_of_features, 1)
    for n in range(1, index):
        projection_matrix = np.c_[projection_matrix, eig_pairs[n][1].reshape(number_of_features, 1)]

    # transformation matrix helps us to reduce an image dimension (actually 28x28 => 764 dimension) to a lower
    # dimensional space by multiplying this matrix
    np_array = np.array(images)
    transformation_matrix = np_array.dot(projection_matrix)

    print("number of samples : ", number_of_samples)
    print("number of features : ", number_of_features)

    print("mean vector : ", mean_vector)
    print("mean matrix : ", mean_matrix)
    print("deviation matrix : ", deviation_matrix)

    print("multiply with transpose : ", multiply_with_transpose)
    print("covariance matrix : ", covariance_matrix)

    print("eigen pairs (sorted) : ", eig_pairs)

    print("Sum of eigen values : ", total)
    print("Explained variance : ", var_exp)
    print("Cumulative explained variance : ", cum_var_exp)

    print("index value : ", index)

    print("projection matrix ", projection_matrix.shape, " : ", projection_matrix)
    print("transformation matrix ", transformation_matrix.shape, " : ", transformation_matrix)

    return projection_matrix, transformation_matrix


def recognize_sudoku(img, bounding_box, dataset, projection_matrix, k_neighbor_value):
    # Crate a empty (dummy) array for beginning
    sudoku_array = [[0 for _ in range(9)] for _ in range(9)]

    resize_dimensions = (640, 480)

    # To make all images standard (because some big pictures makes problem)
    img = cv2.resize(img, resize_dimensions)

    # gray image
    image_sudoku_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    top_left_point_of_rectangle = bounding_box[0]
    bottom_right_point_of_rectangle = bounding_box[1]

    top_right_point_of_rectangle = (bottom_right_point_of_rectangle[0], top_left_point_of_rectangle[1])
    bottom_left_point_of_rectangle = (top_left_point_of_rectangle[0], bottom_right_point_of_rectangle[1])

    # Mask image and eliminate outer space of the sudoku
    (x, y) = image_sudoku_gray.shape
    mask = np.zeros((x, y), np.uint8)

    # I created my own contour here to mask image
    contours = [np.array([[top_left_point_of_rectangle[0], top_left_point_of_rectangle[1]],
                          [top_right_point_of_rectangle[0], top_right_point_of_rectangle[1]],
                          [bottom_right_point_of_rectangle[0], bottom_right_point_of_rectangle[1]],
                          [bottom_left_point_of_rectangle[0], bottom_left_point_of_rectangle[1]]])]

    mask = cv2.drawContours(mask, contours, 0, 255, -1)
    masked = cv2.bitwise_and(mask, image_sudoku_gray)

    rho_threshold = 25  # used for eliminate lines that close each other, remain only 1

    # For each rectangle inside the Sudoku, I calculate average width and height to define its boundaries later
    average_width = (top_right_point_of_rectangle[0] - top_left_point_of_rectangle[0]) / 10
    average_height = (bottom_right_point_of_rectangle[1] - top_right_point_of_rectangle[1]) / 10

    # Blur it
    gaussian_blur = cv2.GaussianBlur(masked, (3, 3), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(gaussian_blur, 50, 20, apertureSize=3)

    # Apply Hough Line Transform, minimum lenght of line is 120 pixels
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)

    if lines is not None:

        # Each line in sudoku will be stored here
        new_lines = []

        # Each point inside the sudoku will be stored here
        points = []

        # Used to eliminate closest lines
        horizontal_rhos = []
        vertical_rhos = []

        # Used to find bottom and right line of the sudoku rectangle
        sudoku_bottom_y = 0
        sudoku_right_x = 0

        for la in lines:

            for rho, theta in la:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                available = 1
                if b > 0.5:
                    # It is horizontal
                    for r in horizontal_rhos:
                        if r - rho_threshold < rho < r + rho_threshold:
                            # Won't add if any line close to this line is already added before
                            available = 0
                    if available == 1:
                        new_lines.append([rho, theta, 0])
                        horizontal_rhos.append(rho)
                        if y0 > sudoku_bottom_y:
                            sudoku_bottom_y = y0
                            sudoku_bottom_line = [rho, theta, 0]

                else:
                    # It is vertical
                    for r in vertical_rhos:
                        if abs(r - rho_threshold) < abs(rho) < abs(r + rho_threshold):
                            # Won't add if any line close to this line is already added before
                            available = 0
                    if available == 1:
                        new_lines.append([rho, theta, 1])
                        vertical_rhos.append(abs(rho))
                        if abs(x0) > sudoku_right_x:
                            sudoku_right_x = abs(x0)
                            sudoku_right_line = [rho, theta, 1]

        cv2.destroyAllWindows()
        for i in range(len(new_lines)):
            if new_lines[i][2] == 0:
                for j in range(len(new_lines)):
                    if new_lines[j][2] == 1:
                        theta1 = new_lines[i][1]
                        theta2 = new_lines[j][1]
                        p1 = new_lines[i][0]
                        p2 = new_lines[j][0]
                        xy = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
                        p = np.array([p1, p2])
                        res = np.linalg.solve(xy, p)
                        points.append(res)

                        # Don't draw rectangle if point is on the bottom line or right line of sudoku rectangle.
                        # Because it won't be inside of the sudoku
                        if new_lines[i] != sudoku_bottom_line and new_lines[j] != sudoku_right_line:

                            x1 = int(res[0] + 5)
                            y1 = int(res[1] + 5)
                            x2 = int(res[0] + int(average_width))
                            y2 = int(res[1] + int(average_height))

                            # Sudoku rectangle
                            cv2.rectangle(gaussian_blur, (x1, y1), (x2, y2), (255, 0, 0), 2)

                            # Crop the sudoku rectangle to use as an input for prediction algorithm
                            crop_rectangle = masked[y1:y2, x1:x2]

                            matrix_i = 9 - int((sudoku_right_x - res[0]) / average_width)
                            matrix_j = 9 - int((sudoku_bottom_y - res[1]) / average_height)

                            # Below controls are here to not get exception if there will be any unexpected value
                            if matrix_i < 0:
                                matrix_i = 0
                            if matrix_i > 8:
                                matrix_i = 8

                            if matrix_j < 0:
                                matrix_j = 0
                            if matrix_j > 8:
                                matrix_j = 8

                            # Below control is again for unexpected values
                            if crop_rectangle.shape[0] > 28 and crop_rectangle.shape[1] > 28:

                                # To compare with MNIST dataset, convert image to 28x28 first
                                cropped_resized = cv2.resize(crop_rectangle, (28, 28))

                                average_value = np.mean(cropped_resized)

                                # Here, I can't solve the problem that how can I choose threshold value for every image
                                # That's why the algorithm cannot predict the number properly most of the time
                                im_bw = cv2.threshold(cropped_resized, average_value - 20, 255, cv2.THRESH_BINARY_INV)[
                                    1]

                                n_white_pix = np.sum(im_bw == 255)

                                # This control added to prevent noisy images predicting false
                                if n_white_pix < 100 or (im_bw.max() - im_bw.min()) == 0:
                                    prediction = 0
                                    sudoku_array[matrix_i][matrix_j] = prediction
                                else:
                                    # Normalize the pixel values between 0 and 1
                                    normalized = (im_bw - im_bw.min()) / (im_bw.max() - im_bw.min())

                                    # Convert (28x28) image to (784,1)
                                    normalized_one_vector = np.ravel(normalized)

                                    # Predict number
                                    prediction = predict_value_knn(dataset,
                                                                   normalized_one_vector.dot(projection_matrix),
                                                                   k_neighbor_value)

                                    print(prediction)

                                    sudoku_array[matrix_i][matrix_j] = prediction

    return sudoku_array


# Returns Sudoku puzzle bounding box following format [(top_left_x, topLefty), (bottom_right_x, bottom_right_y)]
# (You can change the body of this function anyway you like.)
def detect_sudoku(img):
    # variables for the biggest detected contour expected to be Sudoku
    max_contour, max_area, max_per = [], 0, 0

    # smallFrame -> Grayscale -> Blur -> Threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    threshold = cv2.adaptiveThreshold(blur, 255, 0, 1, 11, 7)

    # find all contours in the threshold image
    cnt, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find the largest contour by comparing their size in terms of area and perimeter
    # it also checks the width height ratio of the biggest contour
    if cnt:
        for a in cnt:
            x, y, w, h = cv2.boundingRect(a)
            if cv2.arcLength(a, True) >= max_per and 0.75 < w / h < 1.25 and cv2.contourArea(a) >= max_area:
                max_per = cv2.arcLength(a, True)
                max_contour = a
                max_area = cv2.contourArea(a)

    max_x, max_y, max_w, max_h = cv2.boundingRect(max_contour)

    top_left_x, top_right_y, bot_left_x, bot_left_y = max_x, max_y, max_x + max_w, max_y + max_h
    bounding_box = [(top_left_x, top_right_y), (bot_left_x, bot_left_y)]

    return bounding_box
