"""
K-Nearest Neighbors from scratch using NumPy
الگوریتم K نزدیک‌ترین همسایه برای دسته‌بندی و رگرسیون
"""
import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=5, task="classification"):
        self.k = k
        self.task = task
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _predict_one(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_idx = np.argsort(distances)[: self.k]
        k_labels = self.y_train[k_idx]

        if self.task == "classification":
            most_common = Counter(k_labels).most_common(1)
            return most_common[0][0]
        return np.mean(k_labels)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def score(self, X, y):
        y = np.asarray(y)
        preds = self.predict(X)
        if self.task == "classification":
            return np.mean(preds == y)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
