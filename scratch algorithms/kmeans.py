"""
K-Means Clustering from scratch using NumPy
خوشه‌بندی K-Means با روش k-means++ برای مقداردهی اولیه
"""
import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4, init="k-means++", random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        if self.init == "random":
            idx = rng.choice(n_samples, self.n_clusters, replace=False)
            return X[idx].copy()

        # k-means++: انتخاب هوشمند مراکز اولیه برای همگرایی سریع‌تر
        centroids = [X[rng.integers(n_samples)]]
        for _ in range(1, self.n_clusters):
            dist_sq = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
            )
            probs = dist_sq / dist_sq.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])
        return np.array(centroids)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.centroids = self._init_centroids(X)

        for _ in range(self.max_iters):
            # مرحله ۱: اختصاص هر نقطه به نزدیک‌ترین مرکز
            distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            # مرحله ۲: به‌روزرسانی مراکز
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        self.labels_ = labels
        self.inertia_ = np.sum((X - self.centroids[labels]) ** 2)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
