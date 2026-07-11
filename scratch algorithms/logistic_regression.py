"""
Logistic Regression from scratch using NumPy
رگرسیون لجستیک با گرادیان کاهشی و تابع سیگموید
"""
import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000, l2=0.0):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2  # ضریب regularization
        self.weights = None
        self.bias = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        # کلیپ برای جلوگیری از overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear = X @ self.weights + self.bias
            y_pred = self._sigmoid(linear)

            error = y_pred - y
            dw = (1 / n_samples) * (X.T @ error) + (self.l2 / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            eps = 1e-15
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        linear = X @ self.weights + self.bias
        return self._sigmoid(linear)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)
