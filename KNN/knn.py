import numpy as np
from utility import euclidean_distance
from collections import Counter
class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):

        #Calculate the distance between x and all training datapoints
        distances = [self.euclidean_distances(x, x_train) for x_train in self.X_train]

        #get the k nearest data samples and their labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        #Do the majority voting and choose the most common class label
        most_common_label = Counter(k_nearest_labels).most_common(1)
        return most_common_label[0][0]   ##get most common element from a list of tuple
