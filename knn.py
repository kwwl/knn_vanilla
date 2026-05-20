import math
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, point_a, point_b):
        total = 0

        for index in range(len(point_a)):
            diff = point_a[index] - point_b[index]
            total = total + (diff**2)

        return math.sqrt(total)

    def predict(self, X):
        predictions = []

        for point in X:
            point = list(point)
            all_distances = []
            all_labels = []

            for train_index in range(len(self.X_train)):
                train_point = list(self.X_train[train_index])
                distance = self.euclidean_distance(point, train_point)
                all_distances.append(distance)
                all_labels.append(self.y_train[train_index])

            sorted_indices = sorted(
                range(len(all_distances)), key=lambda position: all_distances[position]
            )
            k_nearest_indices = sorted_indices[: self.k]

            k_nearest_labels = []
            for index in k_nearest_indices:
                k_nearest_labels.append(all_labels[index])

            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)

        return predictions

    def evaluate(self, X, y):
        predictions = self.predict(X)
        true_labels = list(y)
        correct = 0

        for index in range(len(predictions)):
            if predictions[index] == true_labels[index]:

                correct = correct + 1

        accuracy = correct / len(predictions)

        return accuracy
