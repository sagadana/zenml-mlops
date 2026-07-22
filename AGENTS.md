# AGENTS.md — AI Agent Operating Guide

This file describes the project structure, agent personas, available commands, and conventions for AI agents working in this repository.

**Note:** Your responses should be as concise as possible and only include the relevant code snippets. Do not include any explanations or additional context if not requested or necessary.

---

## Project Overview

Unified MLOps orchestration platform built on ZenML. Contains end-to-end ML pipelines deployable locally or on AWS with a single config switch. The **Matrix Factorization (ALS) pipeline** for movie recommendations is the reference implementation and template for all future pipelines.

**Tech stack**: ZenML · ZenML Fan-out/Fan-in (parallel HPO) · Numba (JIT solvers) · ProcessPoolExecutor (ALS training parallelism) · Optuna (HPO) · Evidently AI (monitoring) · FastAPI (serving) · AWS (SageMaker, S3, ECR, DynamoDB) · uv (dependency management)

---

## Repository Structure

Use the following structure as a reference for creating new workflows/pipelines. All workflows are self-contained under `workflows/<workflow_name>/` and share global steps in `steps/`. The `run.py` entrypoint orchestrates all pipelines.

Update this structure as needed, but keep the same conventions for consistency.

```
run.py                                       # Single entrypoint — all pipelines run from here
docker/                                      # Shared Docker assets (all builds use repo root as context)
  pipeline/Dockerfile                        # Base image for all ZenML pipeline steps (shared)
  serving/Dockerfile                         # FastAPI serving image — pass --build-arg WORKFLOW=<name>
  step/Dockerfile.dind                       # DinD image for build_serving_image / deploy_endpoint steps
  zenml/Dockerfile                           # ZenML server (compose)
  ops-db/init.sh                             # MySQL bootstrap for ZenML + Optuna metadata DBs
docker-compose.yml                           # Starts local infra: SeaweedFS, ops-db, ZenML
steps/                                       # Global reusable steps (shared across all workflows)
  monitoring/                                # Drift detection, retrain trigger, log collection
  serving/                                   # Build serving image, deploy endpoint (workflow-agnostic)
workflows/
  matrix_factorization/                       # Self-contained MF pipeline (template for new workflows)
    configs/
      local/                                  # Local dev configs (one YAML per pipeline)
        data_pipeline.yaml
        training_pipeline.yaml                # MovieLens 1M, SQLite HPO
        batch_inference_pipeline.yaml
        deployment_pipeline.yaml
        monitoring_pipeline.yaml
      aws/                                    # AWS production configs (one YAML per pipeline)
        data_pipeline.yaml
        training_pipeline.yaml                # MovieLens 25M, PG HPO
        batch_inference_pipeline.yaml
        deployment_pipeline.yaml
        monitoring_pipeline.yaml
    materializers/                            # Custom ZenML materializers
    models/                                   # Model class definitions
    pipelines/                                # ZenML @pipeline definitions
      data_pipeline.py
      training_pipeline.py
      batch_inference_pipeline.py
      deployment_pipeline.py
      monitoring_pipeline.py
    serving/                                  # FastAPI serving app (app.py + __init__.py)
    steps/                                    # Workflow-specific ZenML @step implementations
      data/                                   # ingest, validate, preprocess
      features/                               # encoders, split, artifacts, select
      hpo/                                    # run_hpo_trial, collect_best_hpo_params
      training/                               # full training loop with checkpoint resume
      evaluation/                             # compute_metrics, register_model
      prediction/                             # batch_predict_user, batch_predict
helpers/                                     # Shared Python utilities (checkpointing, s3_client)
infra/
  local/                                     # Local stack setup script
  aws/                                       # Shared AWS infrastructure scripts
```

---

## Agent Personas

> **Workflow placeholder**: Commands and file paths below use `<workflow_name>` as a placeholder — substitute your workflow's directory name (e.g., `matrix_factorization`). Make targets accept `WORKFLOW=<workflow_name>` to select the active workflow; `<model_name>` is the registered ZenML model name.
>
> **Discover available workflows and pipelines:**
>
> ```bash
> make list-workflows
> make list-pipelines WORKFLOW=<workflow_name>
> ```

### DataEngineer

**Responsibility**: Data ingestion, validation, feature engineering.

**Owned steps**: `ingest_data`, `validate_data`, `preprocess_data`, `build_encoders`, `create_features_artifact`, `load_features_artifact`, `split_data`

**Common commands**:

```bash
# Run training pipeline (includes data ingestion/validation/feature engineering)
make run-local-training WORKFLOW=<workflow_name>

# Build features artifact used by training
make run-local-pipeline WORKFLOW=<workflow_name> PIPELINE=data_pipeline

# Run with caching disabled (force fresh download)
uv run python run.py run --workflow <workflow_name> --pipeline training_pipeline --config workflows/<workflow_name>/configs/local/training_pipeline.yaml --no-cache

# Start local infra services (SeaweedFS, ops-db, ZenML)
docker compose up -d --build
# Inspect artifacts in ZenML dashboard at http://localhost:8237
```

**Files to know**:

- `workflows/<workflow_name>/steps/data/ingest.py` — download/load raw data, returns `pd.DataFrame`
- `workflows/<workflow_name>/steps/data/validate.py` — quality checks, raises `DataValidationError`
- `workflows/<workflow_name>/steps/data/preprocess.py` — dedup, user/item activity filters, top-N per user (`top_ratings_per_user`)
- `workflows/<workflow_name>/steps/features/encoders.py` — entity ID → dense integer index
- `workflows/<workflow_name>/steps/features/artifacts.py` — package/load encoder artifact
- `workflows/<workflow_name>/steps/features/split.py` — temporal stratified train/val/test split

---

### MLEngineer

**Responsibility**: Model training, HPO, evaluation.

**Owned steps**: `run_hpo_trial`, `collect_best_hpo_params`, `train_als_epoch`, `compute_metrics`, `register_model`

**Common commands**:

```bash
# Run training (includes optional HPO controlled by enable_hpo in config)
make run-local-training WORKFLOW=<workflow_name>

# Resume interrupted training (automatic — just re-run the same command)
make run-local-training WORKFLOW=<workflow_name>

# Check checkpoint state
uv run python -c "from helpers.checkpointing import list_checkpoints; print(list_checkpoints('s3://<checkpoint_bucket_name>/<run_id>'))"
```

**Files to know**:

- `workflows/<workflow_name>/models/<algorithm_solver>.py` — JIT-compiled or algorithm-specific solver (e.g. `numba.py`)
- `helpers/checkpointing.py` — `save_checkpoint` / `load_latest_checkpoint` (shared across all workflows)
- `helpers/s3_client.py` — `resolve_zenml_s3_credentials`, `get_s3_client` (shared S3/SeaweedFS helpers)
- `workflows/<workflow_name>/steps/training/train_als.py` — full training step (all epochs + checkpointing) with `ProcessPoolExecutor` partition parallelism and auto-resume
- `workflows/<workflow_name>/steps/hpo/run_hpo.py` — `run_hpo_trial` (single Optuna trial, fan-out) + `collect_best_hpo_params` (fan-in)
- `workflows/<workflow_name>/models/<workflow_name>_model.py` — model class with `predict()` / `batch_predict()`

**HPO Fan-out/Fan-in**:
The training pipeline fans out `hpo_n_trials` independent `run_hpo_trial` steps (one per Optuna trial), then fans in with `collect_best_hpo_params` which reads the best result from the shared Optuna study storage. Parallel execution requires an orchestrator that supports parallel steps (SageMaker, Kubernetes); the local orchestrator runs them sequentially.

```python
# In training_pipeline (simplified):
for i in range(hpo_n_trials):
    trial = run_hpo_trial(trial_idx=i, ..., id=f"hpo_trial_{i}")
    after.append(trial)
best_hyperparams = collect_best_hpo_params(..., after=after)
```

HPO pipeline parameters (configured in `configs/<env>/training_pipeline.yaml`):

- `hpo_n_trials`: Total Optuna trials (local: 20, AWS: 200)
- `hpo_subsample_fraction`: Data fraction per trial (default: 0.2)
- `optuna_storage`: Storage URI (SQLite local, MySQL AWS)
- `optuna_study_name`: Study name (per environment)

**Checkpointing / Resume Protocol**:
The `train_als` step checkpoints after every epoch to `checkpoint_path/<pipeline_run_id>/`:

```
epoch_0001_users.npy   ← user factors after epoch 1
epoch_0001_items.npy   ← item factors after epoch 1
epoch_0001.done        ← written LAST (atomic commit marker)
```

If the step is interrupted, simply re-run the same `make run-local-training WORKFLOW=<workflow_name>` command.
ZenML's step cache will skip all already-completed steps; `train_als` will resume from the last `.done` epoch.

**Hyperparameter search space** (see [workflows/matrix_factorization/steps/hpo/run_hpo.py](workflows/matrix_factorization/steps/hpo/run_hpo.py)):

- `rank`: int [10, 200]
- `regularization`: float log-uniform [1e-3, 10.0]
- `alpha`: float log-uniform [0.01, 10.0]
- `n_iter`: int [5, 25]

**ALS Training Parallelism**:
`train_als` uses `ProcessPoolExecutor` for within-epoch partition-level parallelism (user/item factor updates). Numba handles its own thread-level parallelism via `prange`. The `n_workers` parameter controls the pool size (default: 4).

```
n_workers partition updates per epoch (user) → vstack → user_factors
n_workers partition updates per epoch (item) → vstack → item_factors
```

---

### MLOpsEngineer

**Responsibility**: Infrastructure, stack management, CI/CD, monitoring.

**Owned**: ZenML stacks, infra scripts, monitoring pipeline, retraining triggers.

**Common commands**:

```bash
# Switch between environments (zero code change)
uv run zenml stack set local_docker_stack   # local development
uv run zenml stack set aws_stack     # AWS production

# Verify active stack
uv run zenml stack describe

# Set up AWS infrastructure (idempotent)
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=us-east-1
make infra-aws

# Deploy ZenML server to AWS
uv run zenml deploy --provider aws --region us-east-1

# Run monitoring pipeline
make run-aws-monitoring WORKFLOW=<workflow_name>

# Run full AWS training
uv run zenml stack set aws_stack
make run-aws-training WORKFLOW=<workflow_name>

# View all model versions
uv run zenml model version list <model_name>

# Promote model to production
uv run zenml model version update <model_name> <version> --stage production
```

**Files to know**:

- [infra/local/setup_stacks.sh](infra/local/setup_stacks.sh) — idempotent local stack registration
- [infra/aws/setup_stacks.sh](infra/aws/setup_stacks.sh) — idempotent AWS stack/component registration
- `workflows/<workflow_name>/configs/aws/` — AWS pipeline configs (`training_pipeline.yaml`, `batch_inference_pipeline.yaml`, `deployment_pipeline.yaml`, `monitoring_pipeline.yaml`)
- `steps/monitoring/` — global drift detection, log collection, retrain trigger (shared across all workflows)
- [.github/workflows/ci.yml](.github/workflows/ci.yml) — CI/CD pipeline

**AWS stack components**:
| Component | Name | AWS Service |
|---|---|---|
| Orchestrator | `sagemaker_orchestrator` | SageMaker Pipelines |
| Step Operator | `sagemaker_step_operator` | SageMaker Training/Processing Jobs |
| Artifact Store | `s3_store` | S3 |
| Container Registry | `ecr_registry` | ECR |
| Experiment Tracker | (none) | — |
| Model Registry | (none) | — |
| Data Validator | `evidently_data_validator` | Evidently |

**Local stack components**:
| Component | Name | Backend |
|---|---|---|
| Orchestrator | `local_docker_orchestrator` | Local Docker |
| Artifact Store | `local_s3_store_docker` | SeaweedFS (S3-compatible) |
| Container Registry | `local_container_registry` | Docker registry:2 (`localhost:5001`) |
| Experiment Tracker | (none) | — |
| Model Registry | (none) | — |
| Data Validator | `evidently_data_validator` | Evidently |

---

### ServingEngineer

**Responsibility**: Batch and real-time recommendation serving.

**Owned steps**: `load_als_model`, `get_user_ids`, `get_user_batch_slice`, `predict_user_batch`, `collect_batch_recommendations`, `build_serving_image`, `deploy_endpoint`

**Common commands**:

```bash
# Run deployment pipeline (build Docker image + deploy real-time endpoint)
make run-local-deployment WORKFLOW=<workflow_name>

# Run batch inference pipeline (fan-out user predictions → S3 / DynamoDB)
make run-local-batch-inference WORKFLOW=<workflow_name>

# Health check (once the container is running)
curl http://localhost:8080/health

# Test prediction endpoint
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"user_id": 1, "top_k": 10}'
```

**API reference** (`workflows/<workflow_name>/serving/app.py`):

| Endpoint   | Method | Request Body                 | Response                                                                                                  |
| ---------- | ------ | ---------------------------- | --------------------------------------------------------------------------------------------------------- |
| `/health`  | GET    | —                            | `{status, app_version, model_version, n_users, n_items, rank, cpu_percent, memory_percent, disk_percent}` |
| `/predict` | POST   | `{user_id: int, top_k: int}` | `{user_id, predictions: [{item_id, score}], model_version, latency_ms}`                                   |

**DynamoDB schema** (`movie-recommendations` table):

- Partition key: `id` (String)
- Attribute: `recommendations` (JSON string: `[{item_id, score}, ...]`)
- TTL: `updated_at` (48h from batch job time)

---

## Creating a New Pipeline

Use the `create-e2e-ml-workflow` agent skill (see [.agents/skills/create-e2e-ml-workflow/SKILL.md](.agents/skills/create-e2e-ml-workflow/SKILL.md)).

**Quick summary**:

1. Copy `workflows/matrix_factorization/` to `workflows/<your_workflow_name>/`
2. Update all imports from `workflows.matrix_factorization.` → `workflows.<your_workflow_name>.`
3. `run.py` auto-discovers workflows — no registration needed; verify with `python run.py list-workflows`
4. Create `workflows/<your_workflow_name>/configs/local/{data_pipeline,training_pipeline,batch_inference_pipeline,deployment_pipeline,monitoring_pipeline,online_evaluation_pipeline}.yaml` and `workflows/<your_workflow_name>/configs/aws/{data_pipeline,training_pipeline,batch_inference_pipeline,deployment_pipeline,monitoring_pipeline,online_evaluation_pipeline}.yaml`
5. Unit-test scaffolding is intentionally deferred for now; do not create `tests/` directories until testing is reintroduced.

---

## Keeping Docs and Skills Up to Date

Use the `sync-agent-docs` agent skill (see [.agents/skills/sync-agent-docs/SKILL.md](.agents/skills/sync-agent-docs/SKILL.md)) to keep `AGENTS.md`, skill `SKILL.md` files, stubs, and `setup.sh` files in sync with the actual implementations.

Run it after:

- Any significant change to a reference workflow (`workflows/matrix_factorization`, etc.)
- Adding a new `create-e2e-*` skill
- Reorganising the repository structure

---

## Monitoring & Retraining

The project has two separate monitoring pipelines:

| Pipeline                     | Purpose                                         | Metrics                                                        |
| ---------------------------- | ----------------------------------------------- | -------------------------------------------------------------- |
| `monitoring_pipeline`        | Data Drift & Data Quality (triggers retraining) | DataQualityPreset, DataDriftPreset                             |
| `online_evaluation_pipeline` | Online ranking evaluation (observability only)  | PrecisionTopK, RecallTopK, NDCG, MAP, ScoreDistribution (k=10) |

### monitoring_pipeline

Compares a freshly ingested dataset against the stored training baseline to detect data distribution drift and quality degradation:

```
load_raw_ratings_artifact  → select_features  (comparison / training baseline)
ingest_data(lookback_days) → select_features  (reference  / new data)
evidently_report_step (DataQualityPreset + DataDriftPreset)
check_retrain_trigger
```

`ingest_data` downloads the static MovieLens dataset and simulates recency by shifting timestamps to the present and filtering to the last `lookback_days`. In production, this step would fetch recent ratings from a live data source directly.

Retraining is triggered when drift or data quality thresholds are exceeded, or when the model age exceeds `max_age_days`.

**Drift thresholds** (in `workflows/<workflow_name>/configs/aws/monitoring_pipeline.yaml`):

```yaml
check_retrain_trigger:
  parameters:
    drifted_column_share_threshold: 0.5 # retrain if >50% of columns drift
    missing_values_share_threshold: 0.1 # retrain if >10% missing values
    max_age_days: 30 # retrain if model is >30 days old
```

### online_evaluation_pipeline

Evaluates recommendation quality using Evidently Ranking metrics against recent inference logs:

```
load_raw_ratings_artifact → select_features  (reference / ground-truth ratings)
ingest_logs               → select_features  (current  / model predictions)
evidently_report_step (PrecisionTopK, RecallTopK, NDCG, MAP, ScoreDistribution at k=10)
```

**Manual retrain trigger**:

```bash
make run-aws-training WORKFLOW=<workflow_name>
# or: uv run python run.py run --workflow <workflow_name> --pipeline training_pipeline --config workflows/<workflow_name>/configs/aws/training_pipeline.yaml --stack aws_stack --no-cache
```

---

## Key Conventions

1. **Never use `pipeline` as a variable name** — it shadows the ZenML decorator
2. **All ZenML step outputs must be typed and annotated** — required for artifact tracking
3. **Import pipelines/steps from the module, not from `__init__`** — prevents circular imports in `run.py`
4. **Configs in `configs/<env>/<pipeline>.yaml` control all environment differences** — no code changes needed to switch environments
5. **Checkpoint paths in configs** — `s3://<checkpoint_bucket_name>` for both local (SeaweedFS) and AWS
6. **Global steps live in `steps/`** — cross-workflow steps (e.g. monitoring, serving) go in `steps/<domain>/`; workflow-specific steps go in `workflows/<workflow_name>/steps/`. Import global steps with `from steps.<domain>.<module> import <step>`.
