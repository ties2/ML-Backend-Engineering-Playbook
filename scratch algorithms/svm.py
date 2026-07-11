"""
Support Vector Machine from scratch using NumPy
SVM خطی با بهینه‌سازی Hinge Loss (Gradient Descent) + نسخه kernelized با SMO ساده‌شده
"""
import numpy as np


class LinearSVM:
    """SVM خطی با soft-margin: min ||w||^2/2 + C * sum(hinge_loss)"""

    def __init__(self, lr=0.001, C=1.0, n_iters=1000):
        self.lr = lr
        self.C = C
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        y_ = np.where(y <= 0, -1, 1)  # برچسب‌ها باید {-1, +1} باشند

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iters):
            margins = y_ * (X @ self.w + self.b)
            misclassified = margins < 1

            dw = self.w - self.C * (X[misclassified].T @ y_[misclassified] if misclassified.any() else 0)
            db = -self.C * np.sum(y_[misclassified]) if misclassified.any() else 0.0

            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, 0)

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)


class KernelSVM:
    """
    SVM با کرنل (RBF یا Polynomial) با استفاده از یک نسخه ساده‌شده SMO
    مناسب برای داده‌های غیرخطی و مجموعه‌داده‌های کوچک تا متوسط
    """

    def __init__(self, C=1.0, kernel="rbf", gamma=0.5, degree=3, n_iters=200, tol=1e-3):
        self.C = C
        self.kernel_name = kernel
        self.gamma = gamma
        self.degree = degree
        self.n_iters = n_iters
        self.tol = tol
        self.alphas = None
        self.b = 0.0
        self.X = None
        self.y = None

    def _kernel(self, X1, X2):
        if self.kernel_name == "linear":
            return X1 @ X2.T
        elif self.kernel_name == "poly":
            return (X1 @ X2.T + 1) ** self.degree
        elif self.kernel_name == "rbf":
            X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            sq_dists = X1_sq + X2_sq - 2 * X1 @ X2.T
            return np.exp(-self.gamma * sq_dists)
        else:
            raise ValueError(f"کرنل نامعتبر: {self.kernel_name}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        y = np.where(y <= 0, -1, 1)

        n_samples = X.shape[0]
        self.X, self.y = X, y
        self.alphas = np.zeros(n_samples)
        self.b = 0.0
        K = self._kernel(X, X)

        for _ in range(self.n_iters):
            alpha_changed = 0
            for i in range(n_samples):
                Ei = (self.alphas * self.y) @ K[:, i] + self.b - self.y[i]
                if (self.y[i] * Ei < -self.tol and self.alphas[i] < self.C) or \
                   (self.y[i] * Ei > self.tol and self.alphas[i] > 0):

                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)

                    Ej = (self.alphas * self.y) @ K[:, j] + self.b - self.y[j]
                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]

                    if self.y[i] != self.y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alphas[j] -= self.y[j] * (Ei - Ej) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alphas[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alphas[j])

                    b1 = self.b - Ei - self.y[i] * (self.alphas[i] - alpha_i_old) * K[i, i] \
                         - self.y[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - Ej - self.y[i] * (self.alphas[i] - alpha_i_old) * K[i, j] \
                         - self.y[j] * (self.alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    alpha_changed += 1

            if alpha_changed == 0:
                break

        sv_mask = self.alphas > 1e-5
        self.sv_X = X[sv_mask]
        self.sv_y = y[sv_mask]
        self.sv_alphas = self.alphas[sv_mask]
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        K = self._kernel(X, self.sv_X)
        return K @ (self.sv_alphas * self.sv_y) + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, 0)

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)
