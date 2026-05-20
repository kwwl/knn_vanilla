import math
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y) -> None:
        self.X_train = [list(row) for row in X]
        self.y_train = list(y)

    def euclidean_distance(self, point_a, point_b):
        total = 0

        for index in range(len(point_a)):
            diff = point_a[index] - point_b[index]
            total = total + (diff**2)

        return math.sqrt(total)

    def predict(self, X):
        predictions = []

        for point in X:
            distances = []

            for index in range(len(self.X_train)):
                distance = self.euclidean_distance(point, self.X_train[index])
                distances.append((distance, self.y_train[index]))

            distances.sort(key=lambda element: element[0])

            k_nearest = distances[:self.k]

            labels = []
            for element in k_nearest:
                labels.append(element[1])

            most_common = Counter(labels).most_common(1)[0][0]
            predictions.append(most_common)

        return predictions
