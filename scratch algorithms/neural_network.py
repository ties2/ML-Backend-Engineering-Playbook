"""
Neural Network (Multi-Layer Perceptron) from scratch using NumPy
شبکه عصبی چندلایه با Backpropagation، قابل استفاده برای دسته‌بندی و رگرسیون
"""
import numpy as np


class NeuralNetwork:
    """
    MLP کاملاً متصل با تعداد دلخواه لایه‌ی پنهان
    layer_sizes: مثلا [n_features, 16, 8, n_outputs]
    """

    def __init__(self, layer_sizes, lr=0.01, n_iters=1000, task="classification",
                 activation="relu", random_state=42):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.n_iters = n_iters
        self.task = task  # "classification" (binary) یا "regression" یا "multiclass"
        self.activation_name = activation
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.loss_history = []
        self.X_mean_ = None
        self.X_std_ = None

    # ---------- توابع فعال‌سازی ----------
    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_deriv(z):
        return (z > 0).astype(float)

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _sigmoid_deriv(a):
        return a * (1 - a)

    @staticmethod
    def _softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _init_params(self):
        rng = np.random.default_rng(self.random_state)
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # He initialization برای ReLU
            w = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    def _forward(self, X):
        activations = [X]
        zs = []
        n_layers = len(self.weights)

        for i in range(n_layers):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            zs.append(z)

            if i == n_layers - 1:
                # لایه خروجی
                if self.task == "classification":
                    a = self._sigmoid(z)
                elif self.task == "multiclass":
                    a = self._softmax(z)
                else:  # regression
                    a = z
            else:
                a = self._relu(z) if self.activation_name == "relu" else self._sigmoid(z)
            activations.append(a)

        return activations, zs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # استانداردسازی ورودی برای پایداری و همگرایی بهتر آموزش
        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0)
        self.X_std_[self.X_std_ == 0] = 1.0
        X = (X - self.X_mean_) / self.X_std_

        if y.ndim == 1:
            y = y.reshape(-1, 1) if self.task != "multiclass" else y

        if self.task == "multiclass" and y.ndim == 1:
            n_classes = self.layer_sizes[-1]
            y_onehot = np.zeros((y.shape[0], n_classes))
            y_onehot[np.arange(y.shape[0]), y.astype(int)] = 1
            y = y_onehot

        self._init_params()
        n_samples = X.shape[0]

        for epoch in range(self.n_iters):
            activations, zs = self._forward(X)
            y_pred = activations[-1]

            # محاسبه خطا (loss)
            if self.task == "regression":
                loss = np.mean((y_pred - y) ** 2)
            else:
                eps = 1e-15
                y_pred_c = np.clip(y_pred, eps, 1 - eps)
                loss = -np.mean(np.sum(y * np.log(y_pred_c), axis=1)) if self.task == "multiclass" \
                    else -np.mean(y * np.log(y_pred_c) + (1 - y) * np.log(1 - y_pred_c))
            self.loss_history.append(loss)

            # ---------- Backpropagation ----------
            n_layers = len(self.weights)
            delta = (y_pred - y) / n_samples  # برای MSE، sigmoid+BCE، و softmax+CE مشتق یکسان است

            grads_w = [None] * n_layers
            grads_b = [None] * n_layers

            for i in reversed(range(n_layers)):
                grads_w[i] = activations[i].T @ delta
                grads_b[i] = np.sum(delta, axis=0, keepdims=True)

                if i > 0:
                    da = delta @ self.weights[i].T
                    if self.activation_name == "relu":
                        delta = da * self._relu_deriv(zs[i - 1])
                    else:
                        delta = da * self._sigmoid_deriv(activations[i])

            for i in range(n_layers):
                self.weights[i] -= self.lr * grads_w[i]
                self.biases[i] -= self.lr * grads_b[i]

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        X = (X - self.X_mean_) / self.X_std_
        activations, _ = self._forward(X)
        return activations[-1]

    def predict(self, X):
        proba = self.predict_proba(X)
        if self.task == "classification":
            return (proba >= 0.5).astype(int).flatten()
        elif self.task == "multiclass":
            return np.argmax(proba, axis=1)
        return proba.flatten()

    def score(self, X, y):
        y = np.asarray(y)
        preds = self.predict(X)
        if self.task == "regression":
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot
        return np.mean(preds == y.flatten())
