import pdb
import numpy as np
import code_for_hw3_part2 as hw3


def perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 50
    T = params.get("T", 50)
    (d, n) = data.shape

    theta = np.zeros((d, 1))
    theta_0 = np.zeros((1, 1))
    for t in range(T):
        for i in range(n):
            x = data[:, i : i + 1]
            y = labels[:, i : i + 1]
            if y * hw3.positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook:
                    hook((theta, theta_0))
    return theta, theta_0


# -------------------------------------------------------------------------------
# Auto Data
# -------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data("auto-mpg.tsv")

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features1 = [
    ("cylinders", hw3.raw),
    ("displacement", hw3.raw),
    ("horsepower", hw3.raw),
    ("weight", hw3.raw),
    ("acceleration", hw3.raw),
    ## Drop model_year by default
    ## ('model_year', hw3.raw
    # ),
    ("origin", hw3.raw),
]

features2 = [
    ("cylinders", hw3.one_hot),
    ("displacement", hw3.standard),
    ("horsepower", hw3.standard),
    ("weight", hw3.standard),
    ("acceleration", hw3.standard),
    ## Drop model_year by default
    ## ('model_year', hw3.raw
    # ),
    ("origin", hw3.one_hot),
]

# Construct the standard data and label arrays
auto_data_1, auto_labels_1 = hw3.auto_data_and_labels(auto_data_all, features1)
auto_data_2, auto_labels_2 = hw3.auto_data_and_labels(auto_data_all, features2)
auto_data, auto_labels = auto_data_1, auto_labels_1
features = features1

if False:  # set to True to see histograms
    import matplotlib.pyplot as plt

    for feat in range(auto_data.shape[0]):
        print("Feature", feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat, auto_labels[0, :] > 0])
        plt.hist(auto_data[feat, auto_labels[0, :] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig, (a1, a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat, auto_labels[0, :] > 0])
        a2.hist(auto_data[feat, auto_labels[0, :] < 0])
        plt.show()

# -------------------------------------------------------------------------------
# Analyze auto data
# -------------------------------------------------------------------------------

# Your code here to process the auto data
perceptron = hw3.perceptron
avg_perceptron = hw3.averaged_perceptron
print(
    "\n\n ------------------------------------------------------------------------------ \n\n ANALYSING AUTO DATA (Q4)\n\n"
)
# print('Accuracy for perceptron with features1 T=10: ', hw3.xval_learning_alg(perceptron, auto_data_1, auto_labels_1, 10))
# print('Accuracy for averaged perceptron with features1 T=10: ', hw3.xval_learning_alg(avg_perceptron, auto_data_1, auto_labels_1, 10))
# print('Accuracy for perceptron with features2 T=10: ', hw3.xval_learning_alg(perceptron, auto_data_2, auto_labels_2, 10))
# print('Accuracy for averaged perceptron with features2 T=10: ', hw3.xval_learning_alg(avg_perceptron, auto_data_2, auto_labels_2, 10))
# print('th, th0 for T=10, averaged perceptron with features2: ', hw3.averaged_perceptron(auto_data_2, auto_labels_2, params={'T':10}))
print(
    "\n\n ------------------------------------------------------------------------------ \n"
)

# -------------------------------------------------------------------------------
# Review Data
# -------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data("reviews.tsv")

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(
    *((sample["text"], sample["sentiment"]) for sample in review_data)
)

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)

# -------------------------------------------------------------------------------
# Analyze review data
# -------------------------------------------------------------------------------

# Your code here to process the review data
print(
    "\n ------------------------------------------------------------------------------ \n\n ANALYSING REVIEW DATA (Q5)\n\n"
)

# First let's print out 10-fold cross validation accuracy for both
# perceptron and averaged perceptron with T values (1, 10, 50)

perceptron = hw3.perceptron
avg_perceptron = hw3.averaged_perceptron

# print('Accuracy for perceptron with T=', 50, ': ', hw3.xval_learning_alg(perceptron, review_bow_data, review_labels, 10))
# print('Accuracy for averaged perceptron with T=', 10, ': ', hw3.xval_learning_alg(avg_perceptron, review_bow_data, review_labels, 10))
# Averaged Perceptron with T=10 is most efficient

# th, th0 = hw3.averaged_perceptron(review_bow_data, review_labels, params={'T':10})
# print(th)
# sorted_th_indices = np.argsort(th, axis=0)
# top10_th_indices = sorted_th_indices[:10, 0]
# reverse_dictionary = hw3.reverse_dict(dictionary)
# top10_th = [reverse_dictionary[i] for i in top10_th_indices]
# print('top10_th_indices values: ', th[top10_th_indices])
# print('top10_th_indices: ', top10_th_indices)
# print('top10_th: ', top10_th)

# in review_bow_data we have vectors as columns (19,945 words) and each column denotes a review (10,000 reviews)
# review_labels represents sentiment for each review (10,000 reviews)

# # classifier
# th, th0 = hw3.averaged_perceptron(review_bow_data, review_labels, params={'T':10})
# distances_from_hyperplane = (th.T @ review_bow_data + th0) / np.linalg.norm(th)
# sorted_indices = np.argsort(distances_from_hyperplane)
# print('sorted_indices: ', sorted_indices)
# top10_positive_indices = sorted_indices[0,-1:]
# top10_negative_indices = sorted_indices[0,:1]
# top10_positive_reviews = [review_texts[i] for i in top10_positive_indices]
# top10_negative_reviews = [review_texts[i] for i in top10_negative_indices]
# print('\n\nTop10_positive_reviews: ', top10_positive_reviews)
# print('\n\nTop10_negative_reviews: ', top10_negative_reviews)

# let's sort these reviews i.e columns according to the distance from the classifier

print(
    "\n\n ------------------------------------------------------------------------------ \n\n"
)


# -------------------------------------------------------------------------------
# MNIST Data
# -------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print(
    "mnist_data_all loaded. shape of single images is",
    mnist_data_all[0]["images"][0].shape,
)

# HINT: change the [0] and [1] if you want to access different images
d0 = np.array(mnist_data_all[9]["images"])
d1 = np.array(mnist_data_all[0]["images"])
y0 = np.repeat(-1, len(d0)).reshape(1, -1)
y1 = np.repeat(1, len(d1)).reshape(1, -1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
print(data.shape)
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T


def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    return x.reshape(x.shape[0], -1).T


def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    return np.mean(x, axis=2).T


def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    return np.mean(x, axis=1).T


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    m, n = x.shape[1], x.shape[2]
    top_half = x[:, : m // 2, :]
    bottom_half = x[:, m // 2 :, :]
    return np.vstack(
        (np.mean(top_half, axis=(1, 2)), np.mean(bottom_half, axis=(1, 2)))
    )


# use this function to evaluate accuracy
# acc = hw3.get_classification_accuracy(col_average_features(data), labels)
print(
    "Accuracy for row_average_features: ",
    hw3.get_classification_accuracy(row_average_features(data), labels),
)
print(
    "Accuracy for col_average_features: ",
    hw3.get_classification_accuracy(col_average_features(data), labels),
)
print(
    "Accuracy for top_bottom_features: ",
    hw3.get_classification_accuracy(top_bottom_features(data), labels),
)

# -------------------------------------------------------------------------------
# Analyze MNIST data
# -------------------------------------------------------------------------------

# Your code here to process the MNIST data
