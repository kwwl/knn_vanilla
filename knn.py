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
