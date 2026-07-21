# ML Engineer (Production) Interview Playbook

A structured, chapter-by-chapter playbook for ML Engineer (Production) interviews covering the
Python, backend, database, API, infrastructure, and testing knowledge expected at production-focused
ML/data companies (e.g. fintechs like bunq). Each chapter pairs a source document with an interactive
Jupyter notebook, and the repo also includes a from-scratch ML algorithms library for whiteboard-style
coding rounds.
<p align="center">
  <img src="assets/bbap-sec-ML.png" alt="ML Backend Engineering Playbook" width="60%">
</p>

## Contents

```
.
├── Ch1_Python_for_Production/
├── Ch2_Backend_FastAPI/
├── Ch3_Database/
├── Ch4_API_Design/
├── Ch5_Docker/
├── Ch6_Testing/
├── ML_Engineer_Production_Interview_Prep.docx
├── ml-backend-playbook.md
└── scratch algorithms/
```

## Chapters

Each chapter folder contains the original Word document (`.docx`) and a companion Jupyter notebook
(`.ipynb`) with the same material translated, restructured, and annotated for active study — explanations,
runnable/illustrative code cells, "interview tip" callouts, a model Q&A for the chapter's most-asked
question, and a cheat sheet at the end.

| # | Chapter | Core topics |
|---|---|---|
| 1 | [Python for Production](./Ch1_Python_for_Production/Ch1_Python_for_Production.ipynb) | OOP (encapsulation, ABC vs. Protocol, composition over inheritance), type hints & `mypy`, generators & lazy evaluation, decorators, context managers, `async`/`await`, time & space complexity, clean code & SOLID |
| 2 | [Backend with FastAPI](./Ch2_Backend_FastAPI/Ch2_Backend_FastAPI.ipynb) | Layered architecture (controller → service → repository), FastAPI & Pydantic, dependency injection, the Repository pattern & Unit of Work, the Service pattern, structured error handling, logging & observability, config management, background tasks |
| 3 | [Databases (PostgreSQL)](./Ch3_Database/Ch3_Database.ipynb) | SQL execution order, JOINs, indexing (B-Tree, composite, partial, GIN), transactions & ACID, isolation levels & MVCC, locking (pessimistic/optimistic) & deadlocks, PostgreSQL in production, N+1 / bulk upsert / keyset pagination patterns |
| 4 | [API Design](./Ch4_API_Design/Ch4_API_Design.ipynb) | REST principles & HTTP verb semantics, status codes, JWT authentication, OAuth 2.0 & OIDC, pagination strategies, rate limiting, idempotency, API versioning |
| 5 | [Docker & Containerization](./Ch5_Docker/Ch5_Docker.ipynb) | Images vs. containers vs. VMs, Dockerfile & layer caching, multi-stage builds, image optimization, Docker Compose, networking, volumes & data persistence, production hardening & model-serving considerations |
| 6 | [Testing](./Ch6_Testing/Ch6_Testing.ipynb) | pytest & parametrization, fixtures & `conftest`, mocking & test doubles, unit vs. integration testing, testing async code & FastAPI, coverage, testing ML code & data, CI best practices & flaky tests |

`ML_Engineer_Production_Interview_Prep.docx` is the consolidated source document covering all chapters
in one file. `ml-backend-playbook.md` is a standalone quick-reference summarizing production concerns
(exposing models via APIs, request handling, databases, logging, error handling, scaling, safe
deployment) independent of the chapter series.

> **Note:** `~$_Engineer_Production_Interview_Prep.docx` is a Word lock/temp file (created automatically
> while the `.docx` is open in Microsoft Word). It's safe to delete and isn't part of the actual content.

## Scratch Algorithms

`scratch algorithms/` contains from-scratch (no scikit-learn) NumPy implementations of classic ML
algorithms, useful for the "implement X from scratch" style of interview question:

| File | Algorithm |
|---|---|
| `linear_regression.py` | Linear Regression |
| `logistic_regression.py` | Logistic Regression |
| `knn.py` | k-Nearest Neighbors |
| `naive_bayes.py` | Naive Bayes |
| `decision_tree.py` | Decision Tree |
| `random_forest.py` | Random Forest |
| `gradient_boosting.py` | Gradient Boosting |
| `svm.py` | Support Vector Machine |
| `kmeans.py` | K-Means Clustering |
| `pca.py` | Principal Component Analysis |
| `neural_network.py` | Neural Network (from scratch) |
| `test_algorithms.py` | Tests for the above implementations |

## How to Use This Playbook

1. **Study a chapter:** open the `.ipynb` for the chapter you're preparing. Read each section, run the
   Python code cells where applicable (note: some cells — especially in Ch3/Ch4/Ch5 — contain SQL,
   Dockerfile, or YAML snippets, or reference undefined objects from illustrative examples, and are
   meant for reading rather than execution).
2. **Drill the Q&A:** each section ends with an interview question and a model answer — cover the
   answer and try to say it out loud before checking.
3. **Review the cheat sheet:** every chapter closes with a condensed cheat sheet — use these for a fast
   pass the day of the interview.
4. **Practice coding rounds:** use `scratch algorithms/` to rehearse implementing core ML algorithms
   without library shortcuts, and run `test_algorithms.py` to check your implementations.
5. **Keep the quick-reference handy:** `ml-backend-playbook.md` is a fast lookup for day-to-day
   production engineering concerns, separate from interview prep.

### Running the notebooks

```bash
pip install jupyter
jupyter notebook
```

Then open any `Ch*.ipynb` file. No additional dependencies are required to read the notebooks; a few
code cells (e.g. `jwt`, `redis`, `fastapi`) reference libraries you'd only need installed if you want to
actually execute those specific cells.

## Suggested Study Order

Chapters build on each other conceptually (Python fundamentals → backend architecture → data layer →
API contract → deployment → testing), so working through them in order (1 → 6) is recommended for a
first pass. For a targeted refresher, jump directly to the chapter matching your weak area and start
with its cheat sheet.
