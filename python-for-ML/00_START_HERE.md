# Machine Learning From Foundations to Production

Delivered as **12 runnable Jupyter
notebooks** plus a **50-project catalogue**. Every notebook is already executed, so you can read
it like an article *or* open it and re-run every cell yourself.

The content is organized into four tracks that build on each other:

| Track | Source reference | What you'll be able to do |
|---|---|---|
| **A · Foundations** | *Python with AI* cheat sheet | Manipulate data with NumPy/pandas; understand the math that powers ML |
| **B · Theory** | *ML Handbook* (Radivojac) | Derive and build regression & classifiers from scratch |
| **C · Applied ML** | *Intro to ML with Python* (Müller & Guido) | Use scikit-learn fluently across the full modeling workflow |
| **D · Production** | *MLOps Architecture Reference* (yours) | Ship a model as a tracked, served, monitored service |

---

## The syllabus

### Track A — Foundations
- **`01_python_for_ml`** — NumPy arrays, pandas tables, Python built-ins, `pathlib`, and how to
  prompt an AI assistant for good ML code. *Vectorization timing, groupby, comprehensions.*
- **`02_math_foundations`** — linear algebra (prediction = dot product), probability & the
  Gaussian, **maximum likelihood**, and **gradient descent** — all visualized from scratch.

### Track B — Theory (build it yourself)
- **`03_linear_regression_from_scratch`** — OLS closed form vs gradient descent (they agree),
  the **bias–variance trade-off**, and **regularization** (Ridge/Lasso).
- **`04_linear_classifiers_from_s
- cratch`** — the sigmoid, **logistic regression** trained by
  hand, the linear **decision boundary**, and **naive Bayes**.

### Track C — Applied ML with scikit-learn
- **`05_supervised_learning_1`** — the `fit`/`predict`/`score` workflow, kNN, linear models,
  decision trees, and reading model-complexity curves.
- **`06_supervised_learning_2`** — random forests, gradient boosting, **kernel SVMs**, neural
  nets (MLP), and predicting **probabilities**.
- **`07_unsupervised_learning`** — scaling, **PCA**, **t-SNE**, **k-Means** and **DBSCAN**
  clustering.
- **`08_feature_engineering`** — one-hot encoding, binning, interactions, automatic feature
  selection, and the no-leakage golden rule.
- **`09_model_evaluation`** — cross-validation, **precision/recall/F1**, confusion matrix,
  **ROC-AUC**, grid search, and the imbalanced-data trap.
- **`10_pipelines`** — chaining preprocessing + model into one leak-proof, deployable object;
  `ColumnTransformer`; saving artifacts.
- **`11_text_data`** — bag-of-words, **TF-IDF**, a text classifier, and **LDA** topic modeling.

### Track D — Production
- **`12_mlops_production`** — the reference architecture end to end: project layout, MLflow
  tracking, structured logging, FastAPI + Pydantic serving, **drift monitoring**, Docker/Make/CI.

---

## How to run it

```bash
# 1. Create an environment
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 2. Install the stack used across the notebooks
pip install numpy pandas scikit-learn matplotlib scipy jupyter
#    optional, for notebook 12: pip install mlflow fastapi uvicorn pydantic joblib

# 3. Launch
jupyter notebook        # or: jupyter lab
```

Then open the notebooks **in order** — each assumes the ideas from the previous ones. Read the
markdown, run the cell beneath it, then change a number and run again. You learn this by poking
at it.

> **Note on the datasets.** Everything runs on scikit-learn's built-in datasets (breast cancer,
> wine, digits) and small synthetic data, so no downloads are needed and every cell runs in
> seconds.

---

## A suggested pace

| Week | Notebooks | Focus |
|---|---|---|
| 1 | 01–02 | Get comfortable with the tools and the math intuition |
| 2 | 03–04 | Build models by hand so nothing is a black box |
| 3 | 05–06 | The supervised workhorses in scikit-learn |
| 4 | 07–08 | Unsupervised methods and feature engineering |
| 5 | 09–10 | Evaluate honestly; assemble leak-proof pipelines |
| 6 | 11–12 | Text data, then ship a model to production |

After each notebook, pick a matching project from **`PROJECTS.md`** and build it on the MLOps
architecture. That loop — learn a concept, ship a small system that uses it — is what turns
reading into skill.
