"""
Gaussian Naive Bayes from scratch using NumPy
دسته‌بند بیز ساده با فرض توزیع نرمال برای ویژگی‌ها
"""
import numpy as np


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.mean_ = None
        self.var_ = None
        self.priors_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.mean_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        epsilon = self.var_smoothing * X.var(axis=0).max()

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.mean_[idx] = X_c.mean(axis=0)
            self.var_[idx] = X_c.var(axis=0) + epsilon
            self.priors_[idx] = X_c.shape[0] / X.shape[0]

        return self

    def _log_gaussian(self, class_idx, x):
        mean = self.mean_[class_idx]
        var = self.var_[class_idx]
        numerator = -0.5 * ((x - mean) ** 2) / var
        denominator = -0.5 * np.log(2 * np.pi * var)
        return np.sum(numerator + denominator)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = []
        for x in X:
            posteriors = []
            for idx, c in enumerate(self.classes_):
                log_prior = np.log(self.priors_[idx])
                log_likelihood = self._log_gaussian(idx, x)
                posteriors.append(log_prior + log_likelihood)
            preds.append(self.classes_[np.argmax(posteriors)])
        return np.array(preds)

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)
