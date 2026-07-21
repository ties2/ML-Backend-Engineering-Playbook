# ML Engineer Playbook

A chapter-by-chapter playbook for ML Engineer covering Python, backend,
databases, APIs, infra, testing, distributed systems, ML-in-production, and system design. Each
chapter pairs a source `.docx` with a companion Jupyter notebook (explanations, code, interview
Q&A, cheat sheet).

<p align="center">
  <img src="assets/bbap-sec-ML.png" alt="ML Backend Engineering Playbook" width="60%">
</p>

## Contents

| # | Chapter | Core topics |
|---|---|---|
| 1 | [Python for Production](./Ch1_Python_for_Production/Ch1_Python_for_Production.ipynb) | OOP, type hints, generators, decorators, async, complexity, SOLID |
| 2 | [Backend with FastAPI](./Ch2_Backend_FastAPI/Ch2_Backend_FastAPI.ipynb) | Layered architecture, DI, Repository/Service patterns, error handling, logging |
| 3 | [Databases (PostgreSQL)](./Ch3_Database/Ch3_Database.ipynb) | Indexing, transactions, ACID, isolation levels, locking, pagination |
| 4 | [API Design](./Ch4_API_Design/Ch4_API_Design.ipynb) | REST, JWT/OAuth2, pagination, rate limiting, idempotency, versioning |
| 5 | [Docker](./Ch5_Docker/Ch5_Docker.ipynb) | Images/containers, multi-stage builds, Compose, production hardening |
| 6 | [Testing](./Ch6_Testing/Ch6_Testing.ipynb) | pytest, fixtures, mocking, async/FastAPI testing, CI |
| 7 | [Distributed Systems](./Ch7_Distributed_Systems/Ch7_Distributed_Systems.ipynb) | Redis, queues, Kafka vs. RabbitMQ, delivery guarantees, CAP, resilience |
| 8 | [ML in Production](./Ch8_ML_Production/Ch8_ML_Production.ipynb) | Feature stores, model serving/versioning, monitoring, drift, A/B testing |
| 9 | [System Design](./Ch9_System_Design/Ch9_System_Design.ipynb) | Fraud detection, transaction monitoring (AML), ML pipeline design |

`Doc/ML_Engineer_Production_Interview_Prep.docx` consolidates all chapters in one file.
`ml-backend-playbook.md` is a standalone production-concerns quick reference.

## Other Material

- **`python-for-ML/`** — a separate 12-notebook ML curriculum (foundations → theory-from-scratch →
  scikit-learn → MLOps), see its [start-here guide](./python-for-ML/00_START_HERE.md).
- **`scratch algorithms/`** — NumPy, no-scikit-learn implementations of classic ML algorithms
  (linear/logistic regression, kNN, trees, SVM, k-means, PCA, neural net, etc.) with tests, for
  "implement X from scratch" rounds.
- **`challenges/`** — tiered, hands-on coding challenges (Git, CLI tools, SQL, queues, APIs,
  Docker) as notebooks.
- **`backend-edge-for-ai-engineers.md`** — short essay on why backend skills matter for AI engineers.

## How to Use

1. Work through chapters 1 → 9 in order (each builds on the last); jump to a chapter's cheat sheet
   for a fast pre-interview refresher.
2. Use `scratch algorithms/` and `challenges/` to rehearse hands-on/whiteboard rounds.
3. Keep `ml-backend-playbook.md` handy as a day-to-day production reference.

```bash
pip install jupyter
jupyter notebook   # then open any Ch*.ipynb
```

No extra dependencies are needed to read the notebooks — a few cells (`jwt`, `redis`, `fastapi`)
only need their library installed if you want to actually run them.