"""
تست تمام الگوریتم‌ها روی داده‌های مصنوعی
"""
import numpy as np
from linear_regression import LinearRegression, LinearRegressionNormalEquation
from logistic_regression import LogisticRegression
from kmeans import KMeans
from decision_tree import DecisionTree
from knn import KNN
from naive_bayes import GaussianNaiveBayes
from pca import PCA
from random_forest import RandomForest
from svm import LinearSVM, KernelSVM
from neural_network import NeuralNetwork
from gradient_boosting import GradientBoostingRegressor, GradientBoostingClassifier

np.random.seed(42)



def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


print("=" * 60)
print("1) Linear Regression")
print("=" * 60)
X = np.random.rand(200, 2) * 10
y = 3 * X[:, 0] + 5 * X[:, 1] + 7 + np.random.randn(200) * 0.5
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression(lr=0.01, n_iters=2000).fit(X_train, y_train)
print(f"وزن‌ها: {model.weights}, بایاس: {model.bias:.3f}")
print(f"R^2 (Gradient Descent): {model.score(X_test, y_test):.4f}")

model_ne = LinearRegressionNormalEquation().fit(X_train, y_train)
print(f"ضرایب (Normal Equation): {model_ne.weights}")

print("\n" + "=" * 60)
print("2) Logistic Regression")
print("=" * 60)
X = np.random.randn(300, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = LogisticRegression(lr=0.5, n_iters=1000).fit(X_train, y_train)
print(f"دقت: {clf.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("3) K-Means")
print("=" * 60)
centers = np.array([[0, 0], [5, 5], [0, 5]])
X = np.vstack([c + np.random.randn(100, 2) * 0.5 for c in centers])
km = KMeans(n_clusters=3, random_state=42).fit(X)
print(f"مراکز یافته‌شده:\n{km.centroids}")
print(f"Inertia: {km.inertia_:.4f}")

print("\n" + "=" * 60)
print("4) Decision Tree (Classification)")
print("=" * 60)
X = np.random.randn(300, 4)
y = ((X[:, 0] * X[:, 1] + X[:, 2] > 0)).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y)

tree = DecisionTree(max_depth=5, task="classification").fit(X_train, y_train)
print(f"دقت درخت تصمیم: {tree.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("5) Decision Tree (Regression)")
print("=" * 60)
X = np.random.rand(300, 1) * 10
y = np.sin(X[:, 0]) + np.random.randn(300) * 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y)

tree_reg = DecisionTree(max_depth=6, task="regression").fit(X_train, y_train)
print(f"R^2 درخت رگرسیون: {tree_reg.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("6) KNN")
print("=" * 60)
X = np.random.randn(300, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y)

knn = KNN(k=5, task="classification").fit(X_train, y_train)
print(f"دقت KNN: {knn.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("7) Gaussian Naive Bayes")
print("=" * 60)
nb = GaussianNaiveBayes().fit(X_train, y_train)
print(f"دقت Naive Bayes: {nb.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("8) PCA")
print("=" * 60)
X = np.random.randn(200, 5)
X[:, 1] = X[:, 0] * 2 + np.random.randn(200) * 0.1  # وابستگی مصنوعی
pca = PCA(n_components=2).fit(X)
X_reduced = pca.transform(X)
print(f"شکل داده پس از کاهش بعد: {X_reduced.shape}")
print(f"نسبت واریانس توضیح‌داده‌شده: {pca.explained_variance_ratio_}")

print("\n" + "=" * 60)
print("9) Random Forest (Classification)")
print("=" * 60)
X = np.random.randn(400, 4)
y = ((X[:, 0] * X[:, 1] + X[:, 2] - X[:, 3] > 0)).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y)

rf = RandomForest(n_trees=15, max_depth=6, task="classification", random_state=42).fit(X_train, y_train)
print(f"دقت Random Forest: {rf.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("10) Random Forest (Regression)")
print("=" * 60)
X = np.random.rand(300, 1) * 10
y = np.sin(X[:, 0]) + np.random.randn(300) * 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y)

rf_reg = RandomForest(n_trees=15, max_depth=6, task="regression", random_state=42).fit(X_train, y_train)
print(f"R^2 Random Forest: {rf_reg.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("11) Linear SVM")
print("=" * 60)
X = np.random.randn(300, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y)

svm = LinearSVM(lr=0.001, C=1.0, n_iters=1000).fit(X_train, y_train)
print(f"دقت Linear SVM: {svm.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("12) Kernel SVM (RBF) - داده غیرخطی (دایره‌ای)")
print("=" * 60)
n = 200
r1 = np.random.randn(n // 2) * 0.5
theta1 = np.random.rand(n // 2) * 2 * np.pi
X1 = np.c_[r1 * np.cos(theta1), r1 * np.sin(theta1)]

r2 = 3 + np.random.randn(n // 2) * 0.5
theta2 = np.random.rand(n // 2) * 2 * np.pi
X2 = np.c_[r2 * np.cos(theta2), r2 * np.sin(theta2)]

X = np.vstack([X1, X2])
y = np.array([0] * (n // 2) + [1] * (n // 2))
X_train, X_test, y_train, y_test = train_test_split(X, y)

ksvm = KernelSVM(C=1.0, kernel="rbf", gamma=0.5, n_iters=50).fit(X_train, y_train)
print(f"دقت Kernel SVM (RBF): {ksvm.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("13) Neural Network (Binary Classification)")
print("=" * 60)
X = np.random.randn(300, 2)
y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)  # مرز تصمیم غیرخطی
X_train, X_test, y_train, y_test = train_test_split(X, y)

nn = NeuralNetwork([2, 16, 8, 1], lr=0.1, n_iters=2000, task="classification").fit(X_train, y_train)
print(f"دقت Neural Network: {nn.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("14) Neural Network (Multiclass Classification)")
print("=" * 60)
centers = np.array([[0, 0], [5, 5], [0, 5], [5, 0]])
X = np.vstack([c + np.random.randn(75, 2) * 0.7 for c in centers])
y = np.repeat(np.arange(4), 75)
X_train, X_test, y_train, y_test = train_test_split(X, y)

nn_mc = NeuralNetwork([2, 16, 4], lr=0.1, n_iters=1500, task="multiclass").fit(X_train, y_train)
print(f"دقت Neural Network (چندکلاسه): {nn_mc.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("15) Neural Network (Regression)")
print("=" * 60)
X = np.random.rand(300, 1) * 10
y = np.sin(X[:, 0]) + np.random.randn(300) * 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y)

nn_reg = NeuralNetwork([1, 16, 8, 1], lr=0.05, n_iters=5000, task="regression").fit(X_train, y_train)
print(f"R^2 Neural Network: {nn_reg.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("16) Gradient Boosting Regressor")
print("=" * 60)
X = np.random.rand(300, 1) * 10
y = np.sin(X[:, 0]) + np.random.randn(300) * 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y)

gbr = GradientBoostingRegressor(n_estimators=100, lr=0.1, max_depth=3).fit(X_train, y_train)
print(f"R^2 Gradient Boosting: {gbr.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("17) Gradient Boosting Classifier")
print("=" * 60)
X = np.random.randn(300, 4)
y = ((X[:, 0] * X[:, 1] + X[:, 2] - X[:, 3] > 0)).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y)

gbc = GradientBoostingClassifier(n_estimators=100, lr=0.1, max_depth=3).fit(X_train, y_train)
print(f"دقت Gradient Boosting: {gbc.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("همه ۱۷ الگوریتم با موفقیت اجرا شدند ✅")
print("=" * 60)
