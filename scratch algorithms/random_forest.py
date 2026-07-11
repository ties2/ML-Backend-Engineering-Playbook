"""
Random Forest from scratch using NumPy
جنگل تصادفی: ترکیبی از چند درخت تصمیم با bootstrap sampling و انتخاب تصادفی ویژگی‌ها
"""
import numpy as np
from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2,
                 n_features=None, task="classification", random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # تعداد ویژگی برای هر درخت (پیش‌فرض: sqrt)
        self.task = task
        self.random_state = random_state
        self.trees = []

    def _bootstrap_sample(self, X, y, rng):
        n_samples = X.shape[0]
        idxs = rng.integers(0, n_samples, n_samples)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        n_total_features = X.shape[1]
        if self.n_features is None:
            n_feat_per_tree = max(1, int(np.sqrt(n_total_features)))
        else:
            n_feat_per_tree = self.n_features

        self.trees = []
        for i in range(self.n_trees):
            np.random.seed(rng.integers(0, 1_000_000))  # برای تصادفی بودن انتخاب فیچر داخل درخت
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                task=self.task,
                n_features=n_feat_per_tree,
            )
            X_sample, y_sample = self._bootstrap_sample(X, y, rng)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])  # shape (n_trees, n_samples)

        if self.task == "classification":
            # رأی‌گیری اکثریت برای هر نمونه
            preds = []
            for i in range(X.shape[0]):
                values, counts = np.unique(tree_preds[:, i], return_counts=True)
                preds.append(values[np.argmax(counts)])
            return np.array(preds)
        else:
            return np.mean(tree_preds, axis=0)

    def score(self, X, y):
        y = np.asarray(y)
        preds = self.predict(X)
        if self.task == "classification":
            return np.mean(preds == y)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
