"""
PCA (Principal Component Analysis) from scratch using NumPy
تحلیل مؤلفه‌های اصلی برای کاهش بعد داده
"""
import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        cov_matrix = np.cov(X_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)

        # مرتب‌سازی نزولی بر اساس مقادیر ویژه
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        self.components_ = eigvecs[:, : self.n_components].T
        total_var = eigvals.sum()
        self.explained_variance_ratio_ = eigvals[: self.n_components] / total_var
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
