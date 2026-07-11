"""
Linear Regression from scratch using NumPy
رگرسیون خطی با گرادیان کاهشی (Gradient Descent)
"""
import numpy as np


class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000, fit_intercept=True):
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias

            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            if self.fit_intercept:
                self.bias -= self.lr * db

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias

    def score(self, X, y):
        """R^2 coefficient of determination"""
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


class LinearRegressionNormalEquation:
    """حل تحلیلی با معادله نرمال: w = (X^T X)^-1 X^T y"""

    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.weights
