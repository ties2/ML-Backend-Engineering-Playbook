# ML Backend Engineering Playbook

A practical reference for running ML models in production. Organized by the problems you'll actually hit, with concrete fixes, not just theory.

---

## 1. Exposing Models Through APIs

**Core principle: separate "web server" concerns from "model" concerns.** The API layer should never know how inference works internally — it just calls a predictor interface.

### Do
- **Load the model once at startup**, not per-request. Use a lifespan/startup hook (FastAPI `lifespan`, Flask `before_first_request` equivalent) to load weights into memory and keep a singleton reference.
- **Use FastAPI (or similar) with Pydantic schemas** for request/response validation. Reject malformed input before it touches the model.
- **Version your API** (`/v1/predict`, `/v2/predict`) so you can change output schema without breaking existing clients.
- **Add a `/health` and `/ready` endpoint** — health = process alive, ready = model loaded and warmed up. Kubernetes/load balancers need this distinction.
- **Warm up the model** on startup with a dummy inference call — first real request shouldn't pay cold-start latency (especially GPU kernels, JIT compilation).
- **Return model version + latency in the response metadata.** You will need this for debugging "why did this prediction look weird" six weeks later.
- **Put an API gateway or reverse proxy (nginx/Kong/API Gateway)** in front for auth, rate limiting, and TLS termination — don't build that into your app.

### Avoid
- Loading the model inside the request handler.
- Synchronous blocking calls to a heavy model in an async framework without offloading to a thread/process pool (this stalls the whole event loop).
- Exposing raw stack traces or model internals in error responses.

### Minimal pattern (FastAPI)
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()          # once, at startup
    warmup(model)
    yield
    del model                     # cleanup on shutdown

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    model_version: str
    latency_ms: float

@app.post("/v1/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    ...
```

---

## 2. Handling User Requests

**Core principle: know the difference between fast requests and slow requests, and treat them differently.**

### Fast inference (< 1–2s)
- Handle synchronously, but **enforce a hard timeout** (e.g., 5–10s) so one slow request doesn't hang a worker forever.
- Validate input size/shape early — reject a 50MB image or a 100k-token prompt before it reaches the model.

### Slow inference (> 2s, batch jobs, LLM generation, video processing)
- **Don't make the client wait on an open HTTP connection.** Use one of:
  - **Async job pattern**: client POSTs → you return a `job_id` immediately → client polls `GET /jobs/{id}` or you push a webhook when done.
  - **Message queue**: push the request onto a queue (SQS, RabbitMQ, Kafka, Redis Streams), a worker pool consumes it, result gets written somewhere the client can fetch.
- **Make requests idempotent** — if a client retries because of a timeout, you don't want to run inference twice. Use an idempotency key (client-generated UUID) and dedupe.
- **Apply backpressure**: if your queue is full or GPU workers are saturated, return `429 Too Many Requests` with a `Retry-After` header instead of silently queueing forever.

### Practical checklist
- [ ] Input validation (type, size, encoding) happens before any GPU/CPU work
- [ ] Timeouts set at every layer (client, load balancer, app, model call)
- [ ] Long-running requests use async job or queue pattern
- [ ] Idempotency keys for retryable operations
- [ ] Rate limiting per user/API key

---

## 3. Managing Databases

**Core principle: don't put everything in one database.** ML systems typically need 3–4 different storage types with different access patterns.

| Data type | Storage | Why |
|---|---|---|
| App/user/business data | Postgres/MySQL | Transactional, relational |
| Prediction logs | Data warehouse / object store (S3 + Parquet, BigQuery, Snowflake) | Append-heavy, analytical queries, cheap at scale |
| Feature store (if used) | Redis / DynamoDB / dedicated feature store (Feast, Tecton) | Low-latency key lookups at inference time |
| Vectors/embeddings | Vector DB (pgvector, Pinecone, Weaviate, Qdrant) | Similarity search |

### Do
- **Use connection pooling** (pgbouncer, SQLAlchemy pool) — don't open a new DB connection per request.
- **Run migrations through a tool** (Alembic, Flyway) — never hand-edit schema in prod.
- **Add read replicas** once read traffic (e.g., dashboards, analytics on prediction logs) competes with write traffic (live inference logging).
- **Index on what you actually query** — `model_version`, `timestamp`, `user_id` are the common filters for prediction logs; index those, not everything.
- **Separate hot path from analytics path.** Don't let a dashboard query against your live prediction-logging table block inference writes — stream logs to a warehouse asynchronously instead.

### Avoid
- Writing prediction logs synchronously in the request path if it can be avoided — use a background task or a queue so a slow DB write doesn't add latency to inference.
- Storing large binary blobs (images, raw model outputs) directly in a relational DB — put them in object storage (S3/GCS) and store the reference/URL in the DB.

---

## 4. Logging Predictions

**Core principle: you can't debug or retrain what you didn't log.** But you also can't log everything at full fidelity forever — plan for volume.

### What to log per prediction
- Request ID / correlation ID (trace it across services)
- Timestamp
- Model name + version (critical — "which model produced this?" is the #1 debugging question)
- Input (or a hash/reference if input is large or sensitive)
- Output + confidence/probability scores
- Latency (inference time, total request time)
- User/session ID (if applicable, respecting privacy rules)

### Do
- **Use structured logging** (JSON logs), not free-text — so you can query/filter later.
- **Log asynchronously** — fire-and-forget to a queue or buffer, don't block the response on a logging write.
- **Include correlation IDs** that thread through the whole request (API → queue → worker → DB) so you can reconstruct a single request's journey.
- **Sample at scale.** At high QPS, logging 100% of full inputs/outputs gets expensive fast. Log 100% of metadata (latency, version, status) but sample full payloads (e.g., 1–10%) or log full payloads only for errors/low-confidence predictions.
- **Redact or hash PII** before it hits logs — don't let logging become a compliance problem.
- **Feed prediction logs into a monitoring pipeline** to catch data drift and performance decay over time (compare live input distribution vs training distribution).

### Avoid
- Logging only when something breaks — you need a baseline of "normal" to know what "broken" looks like.
- Mixing debug/print statements with structured production logs.

---

## 5. Handling Errors and Failures

**Core principle: classify failures, then design a specific response for each class.** "Something went wrong" is not a strategy.

### Classify errors
1. **Client errors** (bad input, malformed request) → `4xx`, return a clear validation message.
2. **Model errors** (inference fails, OOM, NaN output, unexpected shape) → log full context, return `5xx` or a fallback response, alert if rate spikes.
3. **Infrastructure errors** (DB down, timeout, dependency unavailable) → retry with backoff, circuit-break, degrade gracefully.

### Do
- **Retry transient failures with exponential backoff + jitter** (e.g., DB connection blips, downstream service timeouts). Cap retry attempts.
- **Use a circuit breaker** for dependencies that can fail hard (external APIs, a flaky feature store) — stop hammering a dead service and fail fast instead.
- **Have a fallback response** where possible: a simpler heuristic, a cached previous prediction, or a "default" answer, rather than a raw 500 — especially for user-facing features.
- **Set timeouts at every hop** — client→gateway, gateway→app, app→model, app→DB. An untimed call anywhere is a latent outage.
- **Alert on error *rate*, not raw count** — 5 errors out of 10 requests is very different from 5 out of 500,000.
- **Return actionable error responses** to clients: error code, human message, request ID to reference when they contact support.

### Avoid
- Silently swallowing exceptions and returning a default value with no logging — this hides real problems.
- Infinite retry loops with no backoff (this is how you DDoS your own dependency during an incident).

---

## 6. Scaling for Multiple Users

**Core principle: make the service stateless and scale it horizontally; make the model serving layer efficient before you throw more hardware at it.**

### Application layer
- **Keep app servers stateless** — no in-memory session state — so you can add/remove instances freely behind a load balancer.
- **Autoscale on meaningful metrics**: queue depth or request latency, not just raw CPU (GPU-bound inference often has low CPU usage but is fully saturated).

### Model serving layer
- **Batch requests** where latency budget allows — most frameworks (TorchServe, Triton, vLLM for LLMs) support dynamic batching, which dramatically improves GPU throughput.
- **Use an optimized runtime** — ONNX Runtime, TensorRT, or quantization (int8/fp16) can cut inference time significantly with minimal accuracy loss.
- **Separate the model-serving service from the API service** so you can scale GPU workers independently from lightweight API pods.
- **Cache identical/repeated requests** (Redis with a hash of the input as key) if your traffic has repeat queries — common in search/recommendation.

### Traffic management
- **Use a queue to absorb spikes** instead of scaling instantly to match peak load — smooths cost and protects against overload.
- **Load balance across model replicas** with awareness of GPU memory/queue depth, not just round robin.

### Checklist
- [ ] App servers are stateless and horizontally scalable
- [ ] Model server uses batching/optimized runtime
- [ ] Autoscaling triggers on latency/queue depth, not just CPU
- [ ] Caching layer for repeated queries
- [ ] Load test before you need to (know your breaking point in advance)

---

## 7. Deploying Updates Safely

**Core principle: never let a new model version reach 100% of traffic without evidence it's at least as good as the old one.**

### Deployment strategies (pick based on risk tolerance)
- **Shadow deployment**: new model runs alongside the old one on real traffic, but its output is only logged/compared, never returned to users. Best first step for a risky model change.
- **Canary release**: route a small % of traffic (1–5%) to the new version, monitor error rate/latency/business metrics, ramp up gradually.
- **Blue-green deployment**: run old (blue) and new (green) versions in parallel, switch traffic all at once, keep blue ready for instant rollback.
- **A/B testing**: split traffic by user cohort to compare business-metric impact (not just technical correctness) between model versions.

### Do
- **Version and register every model** (MLflow, a model registry, or even a simple S3-path-with-metadata convention) — you need to know exactly which artifact is live at all times.
- **Automate rollback** — if error rate or a key metric crosses a threshold post-deploy, revert automatically, don't wait for a human to notice.
- **Run a smoke test suite** against the new model before it takes real traffic — a handful of known inputs with expected output ranges.
- **Decouple code deploys from model deploys** — you should be able to update model weights without redeploying application code, and vice versa (load model from a registry/path, not baked into the container image).
- **Keep the previous version warm** for a rollback window instead of tearing it down immediately after a new deploy.

### Avoid
- Deploying a new model directly to 100% of traffic on a Friday afternoon with no monitoring window.
- Coupling model artifact updates to full CI/CD app redeploys — makes rollback slower and riskier than it needs to be.

---

## Quick-Reference Summary

| Area | #1 Rule |
|---|---|
| Exposing APIs | Load model once at startup, never per-request |
| Handling requests | Slow inference → async job/queue, not a blocking HTTP call |
| Databases | Different data types need different stores; don't log predictions into your transactional DB synchronously |
| Logging | Log model version + correlation ID on every prediction, always |
| Errors | Classify failures, add timeouts everywhere, retry with backoff |
| Scaling | Batch inference + stateless app layer + autoscale on latency/queue depth |
| Deployment | Shadow/canary before full rollout, automate rollback |

---

*This playbook is a starting framework — adapt thresholds (timeout values, canary %, sampling rates) to your actual traffic and risk tolerance.*
