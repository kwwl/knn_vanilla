import math
from collections import Counter


class KNN:
    def __init__(self, k: int = 3):
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
