"""
Gradient Boosting from scratch using NumPy
تقویت گرادیانی: ساخت متوالی درخت‌های رگرسیون کوچک برای تصحیح خطای باقیمانده
"""
import numpy as np
from decision_tree import DecisionTree


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, lr=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.init_pred = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.init_pred = np.mean(y)
        residual = y - self.init_pred
        self.trees = []

        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 task="regression")
            tree.fit(X, residual)
            update = tree.predict(X)
            residual -= self.lr * update
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        y_pred = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            y_pred += self.lr * tree.predict(X)
        return y_pred

    def score(self, X, y):
        y = np.asarray(y)
        preds = self.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


class GradientBoostingClassifier:
    """دسته‌بندی دودویی با Gradient Boosting و لاجیت (log-odds) به‌عنوان تابع هدف"""

    def __init__(self, n_estimators=100, lr=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.init_log_odds = None

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.init_log_odds = np.log(p / (1 - p))
        F = np.full(y.shape[0], self.init_log_odds)
        self.trees = []

        for _ in range(self.n_estimators):
            p_pred = self._sigmoid(F)
            residual = y - p_pred  # گرادیان منفی برای log-loss

            tree = DecisionTree(max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 task="regression")
            tree.fit(X, residual)
            update = tree.predict(X)
            F += self.lr * update
            self.trees.append(tree)

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.init_log_odds)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return self._sigmoid(F)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)
