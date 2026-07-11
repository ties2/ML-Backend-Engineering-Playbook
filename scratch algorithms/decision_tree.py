"""
Decision Tree (Classification & Regression) from scratch using NumPy
درخت تصمیم برای دسته‌بندی (با معیار Gini) و رگرسیون (با معیار MSE)
"""
import numpy as np


class _Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # فقط برای برگ‌ها پر می‌شود

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, task="classification", n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task = task  # "classification" یا "regression"
        self.n_features = n_features  # برای random forest کاربرد دارد
        self.root = None

    # ---------- معیارهای ناخالصی ----------
    @staticmethod
    def _gini(y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)

    @staticmethod
    def _mse(y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _impurity(self, y):
        return self._gini(y) if self.task == "classification" else self._mse(y)

    # ---------- ساخت درخت ----------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_features_total = X.shape[1]
        if self.n_features is None:
            self.n_features = self.n_features_total
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _grow_tree(self, X, y, depth):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # شرایط توقف
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return _Node(value=self._leaf_value(y))

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh, best_gain = self._best_split(X, y, feat_idxs)

        if best_gain is None or best_gain <= 0:
            return _Node(value=self._leaf_value(y))

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return _Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        parent_impurity = self._impurity(y)

        for feat_idx in feat_idxs:
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                n = len(y)
                n_l, n_r = left_mask.sum(), right_mask.sum()
                imp_l = self._impurity(y[left_mask])
                imp_r = self._impurity(y[right_mask])
                weighted_impurity = (n_l / n) * imp_l + (n_r / n) * imp_r
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh, (best_gain if split_idx is not None else None)

    def _leaf_value(self, y):
        if self.task == "classification":
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]
        return np.mean(y)

    # ---------- پیش‌بینی ----------
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def score(self, X, y):
        y = np.asarray(y)
        preds = self.predict(X)
        if self.task == "classification":
            return np.mean(preds == y)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
