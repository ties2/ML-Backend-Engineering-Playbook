# 50 Machine Learning Projects — on your MLOps Architecture

Every project below is meant to be built **inside your reference architecture** (`FastAPI` +
`MLflow` + `DVC` + `Docker` + `Ruff` + `GitHub Actions` + `Make`). Rather than repeat the whole
stack 50 times, here's the shared template — then each project only notes what changes.

## The shared blueprint (applies to all 50)

```
project-name/
├── .dvc/                      # raw + processed data versioned here
├── .github/workflows/ci-cd.yaml
├── Makefile                   # make train | make serve | make docker | make lint
├── pyproject.toml             # Ruff + Pytest config
├── requirements.txt
├── data/                      # DVC-tracked, Git-ignored
├── src/
│   ├── utils/logger.py        # per-module structured logs
│   ├── features/              # feature engineering
│   ├── models/train.py        # trains → logs params/metrics/model to MLflow
│   ├── models/evaluate.py
│   ├── serving/app.py         # FastAPI: /health + /predict
│   ├── serving/schemas.py     # Pydantic request/response validation
│   └── monitoring/drift_detector.py
└── deployment/Dockerfile
```

**The loop is always the same:** version data with DVC → train and log to MLflow → save the
fitted pipeline as the artifact → serve it behind a validated FastAPI endpoint → monitor for
drift → retrain when drift or metrics degrade. Per project, you're really deciding four things:

- **Data** — what you version in DVC.
- **Model** — what `train.py` fits.
- **Endpoint** — what `/predict` accepts and returns.
- **Monitor** — the signal `drift_detector.py` watches, and the retrain trigger.

Projects are grouped into five tiers of increasing difficulty. Do them in roughly this order,
pairing each with the tutorial notebook it exercises.

---

## Tier 1 — Foundations (tabular, clean data)
*Pairs with notebooks 05, 08, 09. Goal: get the whole loop working on easy data.*

1. **Iris / Wine species classifier** — multiclass classification on a classic clean dataset.
   *Endpoint:* 4–13 numeric features → class + probabilities. *Monitor:* feature-mean drift.

2. **Titanic survival predictor** — binary classification with mixed numeric/categorical fields.
   *Model:* `ColumnTransformer` + gradient boosting. *Track:* which features mattered most.

3. **House price regression** — predict a continuous price from property features.
   *Model:* Ridge/GradientBoostingRegressor. *Metric:* MAE + RMSE logged per run. *Monitor:*
   prediction-distribution shift as neighborhoods change.

4. **Diabetes progression regression** — regression on medical measurements.
   *Focus:* baseline vs regularized models; log R² and residual plots as MLflow artifacts.

5. **Wine quality scorer** — ordinal-ish regression/classification on physicochemical tests.
   *Endpoint:* returns a quality score 0–10. *Monitor:* input range checks in Pydantic.

6. **Breast-cancer diagnostic** — the notebook's running example, productionized.
   *Focus:* threshold tuning for high recall; expose the threshold as a config value.

7. **Mushroom edibility classifier** — all-categorical features → binary safe/poisonous.
   *Model:* one-hot + logistic regression. *Focus:* interpretable coefficients in the response.

8. **Bank term-deposit acceptance** — will a customer subscribe? (imbalanced).
   *Focus:* class weighting; log precision/recall not accuracy. *Monitor:* base-rate drift.

9. **Student pass/fail predictor** — classification from study + demographic features.
   *Focus:* fairness slice-metrics logged per subgroup in MLflow.

10. **Car MPG / emissions regression** — predict fuel economy from engine specs.
    *Focus:* feature interactions (notebook 08); polynomial features in the pipeline.

---

## Tier 2 — Applied classification & regression (real-world messiness)
*Pairs with notebooks 06, 08, 10. Goal: handle missing data, imbalance, and tuning.*

11. **Credit-default risk scorer** — predict loan default probability.
    *Endpoint:* returns calibrated risk + reason codes. *Monitor:* population-stability index.

12. **Customer churn predictor** — flag subscribers likely to leave.
    *Focus:* GridSearchCV over the whole pipeline; log the best params. *Monitor:* feature drift
    as product usage evolves.

13. **Insurance claim-cost regression** — predict payout amounts (long-tailed target).
    *Focus:* log-transform the target; track error on the tail separately.

14. **Employee attrition model** — HR analytics classification.
    *Focus:* SHAP-style feature attributions saved as run artifacts.

15. **Hospital readmission risk** — 30-day readmission prediction.
    *Focus:* strict no-leakage pipeline (notebook 10); recall-oriented thresholds.

16. **Flight-delay predictor** — classify/regress delay from schedule + weather features.
    *Monitor:* seasonal drift; scheduled monthly retrain via a GitHub Actions cron.

17. **E-commerce purchase-intent** — will this session convert?
    *Endpoint:* scores a session in real time. *Focus:* latency budget; load model once at startup.

18. **Energy-consumption forecaster (tabular)** — predict next-hour load from calendar + weather.
    *Focus:* time-aware train/test split (no shuffling). *Monitor:* concept drift on error.

19. **Fraud-transaction detector** — highly imbalanced binary classification.
    *Focus:* precision-recall AUC over ROC-AUC; anomaly-aware thresholding. *Monitor:* fraud-rate
    drift with alerting.

20. **Real-estate rent estimator** — regression with geospatial + categorical features.
    *Focus:* target encoding for high-cardinality location; DVC-version the geo lookup table.

---

## Tier 3 — Unstructured data: text, images, time series
*Pairs with notebooks 07, 11. Goal: vectorize non-tabular inputs and serve them.*

21. **Sentiment analysis API** — classify review text as positive/negative.
    *Model:* TF-IDF + logistic regression. *Endpoint:* raw text in, label + confidence out.

22. **Spam / phishing filter** — binary text classification.
    *Focus:* character + word n-grams. *Monitor:* vocabulary drift (new spam patterns).

23. **News topic classifier** — route articles into categories.
    *Model:* TF-IDF + linear SVM. *Focus:* multiclass metrics; per-class confusion analysis.

24. **Support-ticket router** — classify incoming tickets to the right team.
    *Endpoint:* returns top-3 teams with probabilities for human-in-the-loop routing.

25. **Resume–job matcher** — rank resumes against a job description (text similarity).
    *Model:* TF-IDF cosine similarity → learned re-ranker. *Focus:* offline ranking metrics.

26. **Topic discovery dashboard** — unsupervised LDA over a document corpus.
    *Focus:* no labels; serve topic assignments; DVC-version the corpus snapshots.

27. **Handwritten-digit recognizer** — image classification on `digits`/MNIST-style data.
    *Model:* MLP or small CNN. *Endpoint:* accepts a flattened image array.

28. **Image quality / blur detector** — classify images as sharp vs blurry.
    *Focus:* engineered image features (variance of Laplacian) + classifier.

29. **Time-series anomaly detector** — flag unusual points in sensor streams.
    *Model:* rolling stats + IsolationForest. *Monitor:* the anomaly rate itself.

30. **Demand forecaster** — predict next-period sales per SKU.
    *Focus:* lag/rolling features; walk-forward validation; per-SKU error tracking in MLflow.

---

## Tier 4 — Advanced systems (ranking, recommendation, optimization)
*Goal: systems with more than one model or a feedback loop.*

31. **Movie recommender** — collaborative filtering (matrix factorization).
    *Endpoint:* user_id → top-N items. *Monitor:* catalog coverage + cold-start rate.

32. **Product recommender ("also bought")** — item–item similarity from co-purchase data.
    *Focus:* nightly batch recompute (`make train`) + fast lookup serving.

33. **Search result re-ranker** — learning-to-rank over query/document features.
    *Metric:* NDCG logged per run. *Focus:* pairwise training data from click logs (DVC-versioned).

34. **Dynamic pricing model** — predict demand at candidate prices to optimize revenue.
    *Focus:* two models (demand + revenue curve); guardrails in the serving layer.

35. **Ad click-through-rate predictor** — high-cardinality categorical CTR model.
    *Focus:* hashing/embedding features; sub-100ms serving; heavy monitoring on calibration.

36. **Multi-armed-bandit content picker** — online learning that balances explore/exploit.
    *Focus:* stateful serving; log reward per arm; A/B against a static baseline.

37. **Customer lifetime-value predictor** — regression feeding a marketing budget optimizer.
    *Focus:* chained pipeline; log both model error and downstream business metric.

38. **Anomaly-based network intrusion detector** — unsupervised + supervised hybrid.
    *Monitor:* drift on traffic distribution; auto-retrain trigger on drift alarm.

39. **Predictive-maintenance model** — predict time-to-failure from equipment telemetry.
    *Focus:* survival-style target; alerting endpoint; DVC-versioned sensor archives.

40. **Portfolio-risk classifier** — categorize assets by risk from market features.
    *Focus:* strict temporal validation; model card auto-generated as an MLflow artifact.

---

## Tier 5 — Production-grade capstones (the full loop, for real)
*Goal: exercise every part of the architecture — tracking, CI/CD, drift, rollback, A/B.*

41. **End-to-end fraud platform** — streaming scoring with a feature store and drift alarms.
    *Focus:* the complete reference stack; canary + stable model versions behind `/predict`.

42. **A/B model-serving harness** — route traffic between two model versions by user hash.
    *Focus:* the reference `app.py` routing pattern; log which version served each request;
    compare live metrics in MLflow.

43. **Automated retraining pipeline** — drift detector triggers `make train` via GitHub Actions.
    *Focus:* close the loop: monitor → retrain → evaluate → auto-promote if better, else hold.

44. **Model registry + rollback** — promote/demote models through staging→production in MLflow.
    *Focus:* one-command rollback; CI gate that blocks promotion on metric regression.

45. **Feature-store-backed service** — share engineered features across training and serving.
    *Focus:* eliminate train/serve skew; DVC-version feature definitions.

46. **Shadow-deployment evaluator** — run a new model alongside prod without affecting users.
    *Focus:* log predictions from both; compare offline before any traffic switch.

47. **Batch + online hybrid** — nightly batch scores everyone; online endpoint handles new users.
    *Focus:* two serving paths sharing one artifact; reconciliation monitoring.

48. **Multi-model ensemble service** — serve a stacked ensemble behind one endpoint.
    *Focus:* latency vs accuracy trade-off; per-sub-model health checks.

49. **Explainability API** — `/predict` returns the prediction *and* per-feature attributions.
    *Focus:* Pydantic response schema with reason codes; audit-log every explanation.

50. **Full observability stack** — the capstone: metrics, structured logs, drift, and alerts on
    one dashboard, with automated retraining and A/B promotion.
    *Focus:* everything above, wired together — this is a portfolio centerpiece.

---

## How to pick your next project

- **New to the loop?** Start at #1–3 and get *one* model fully through train → serve → monitor
  before optimizing anything.
- **Comfortable with modeling, weak on production?** Jump to Tier 5 and rebuild a simple model
  you already know, but wire up MLflow, Docker, CI, and drift properly.
- **Building a portfolio?** One project from each tier (e.g. #6, #12, #21, #31, #42) shows the
  full range: clean data, messy data, unstructured data, a system, and a production platform.

Each project reuses the same architecture, so your second project is faster than your first, and
your tenth is mostly a new `features/` module and a new `/predict` schema. That compounding is
the point.
