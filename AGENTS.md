# AGENTS.md — AI Agent Operating Guide

This file describes the project structure, agent personas, available commands, and conventions for AI agents working in this repository.

---

## Project Overview

Unified MLOps orchestration platform built on ZenML. Contains end-to-end ML pipelines deployable locally or on AWS with a single config switch. The **Matrix Factorization (ALS) pipeline** for movie recommendations is the reference implementation and template for all future pipelines.

**Tech stack**: ZenML · Dask (distributed training) · Numba (JIT solvers) · Optuna (HPO) · Evidently AI (monitoring) · FastAPI (serving) · MLflow (experiment tracking) · AWS (SageMaker, S3, ECR, DynamoDB) · uv (dependency management)

---

## Repository Structure

```
run.py                                       # Single entrypoint — all pipelines run from here
steps/                                       # Global reusable steps (shared across all workflows)
  monitoring/                                # Drift detection, retrain trigger, log collection
workflows/
  matrix_factorization/                       # Self-contained MF pipeline (template for new workflows)
    configs/
      local.yaml                              # Local dev config (MovieLens 1M, SQLite HPO)
      aws.yaml                                # AWS production config (MovieLens 25M, PG HPO)
    materializers/                            # Custom ZenML materializers
    models/                                   # Model class definitions
    pipelines/                                # ZenML @pipeline definitions
    serving/                                  # FastAPI serving app + Dockerfile
    steps/                                    # Workflow-specific ZenML @step implementations
    tests/unit/                               # Unit tests
    utils/                                    # MF-specific utilities (ALS solvers, checkpointing, Dask)
    Dockerfile                                # ZenML step base image
    README.md
infra/aws/                                   # Shared AWS infrastructure scripts
```

---

## Agent Personas

> **Workflow placeholder**: Commands and file paths below use `<workflow_name>` as a placeholder — substitute your workflow's directory name (e.g., `matrix_factorization`). Make targets accept `WORKFLOW=<workflow_name>` to select the active workflow; `<model_zenml_name>` is the registered ZenML model name.
>
> **Discover available workflows and pipelines:**
>
> ```bash
> make list-workflows
> make list-pipelines WORKFLOW=<workflow_name>
> ```

### DataEngineer

**Responsibility**: Data ingestion, validation, feature engineering.

**Owned steps**: `ingest_data`, `validate_data`, `build_encoders`, `split_data`

**Common commands**:

```bash
# Run data pipeline locally
make run-local-data WORKFLOW=<workflow_name>

# Run with caching disabled (force fresh download)
uv run python run.py run --workflow <workflow_name> --pipeline data --config workflows/<workflow_name>/configs/local.yaml --no-cache

# Inspect artifacts in ZenML dashboard
uv run zenml up  # starts local dashboard at http://localhost:8237
```

**Files to know**:

- `workflows/<workflow_name>/steps/data_ingestion/ingest.py` — download/load raw data + Dask partitioning
- `workflows/<workflow_name>/steps/data_validation/validate.py` — quality checks, raises `DataValidationError`
- `workflows/<workflow_name>/steps/feature_engineering/encoders.py` — entity ID → dense integer index
- `workflows/<workflow_name>/steps/feature_engineering/split.py` — temporal stratified train/val/test split
- `workflows/<workflow_name>/materializers/dask_dataframe_materializer.py` — Parquet serialization

---

### MLEngineer

**Responsibility**: Model training, HPO, evaluation.

**Owned steps**: `run_hpo`, `train_als`, `compute_metrics`, `register_model`

**Common commands**:

```bash
# Run HPO (optional, updates Optuna study)
make run-local-hpo WORKFLOW=<workflow_name>

# Run training (with HPO disabled, using default hyperparams)
make run-local-training WORKFLOW=<workflow_name>

# Resume interrupted training (automatic — just re-run the same command)
make run-local-training WORKFLOW=<workflow_name>

# Check checkpoint state
uv run python -c "from helpers.checkpointing import list_checkpoints; print(list_checkpoints('./checkpoints/<run_id>'))"
```

**Files to know**:

- `workflows/<workflow_name>/utils/<algorithm_solver>.py` — JIT-compiled training solver (algorithm-specific)
- `helpers/checkpointing.py` — `save_checkpoint` / `load_latest_checkpoint` (shared across all workflows)
- `workflows/<workflow_name>/steps/training/train.py` — distributed training loop with checkpointing
- `workflows/<workflow_name>/steps/hpo/run_hpo.py` — Optuna + Dask distributed HPO
- `workflows/<workflow_name>/models/<workflow_name>_model.py` — model class with `predict()` / `batch_predict()`

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

---

### MLOpsEngineer

**Responsibility**: Infrastructure, stack management, CI/CD, monitoring.

**Owned**: ZenML stacks, infra scripts, monitoring pipeline, retraining triggers.

**Common commands**:

```bash
# Switch between environments (zero code change)
uv run zenml stack set local_stack   # local development
uv run zenml stack set aws_stack     # AWS production

# Verify active stack
uv run zenml stack describe

# Set up AWS infrastructure (idempotent)
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=us-east-1
export ZENML_EXECUTION_ROLE_ARN=arn:aws:iam::123456789012:role/zenml-execution-role
make infra-setup

# Deploy ZenML server to AWS
uv run zenml deploy --provider aws --region us-east-1

# Run monitoring pipeline
make run-aws-monitoring WORKFLOW=<workflow_name>

# Run full AWS training
uv run zenml stack set aws_stack
make run-aws-training WORKFLOW=<workflow_name>

# View all model versions
uv run zenml model version list <model_zenml_name>

# Promote model to production
uv run zenml model version update <model_zenml_name> <version> --stage production
```

**Files to know**:

- [infra/aws/setup_stacks.sh](infra/aws/setup_stacks.sh) — idempotent ZenML stack registration
- [infra/aws/iam_policy.json](infra/aws/iam_policy.json) — least-privilege IAM policy
- `workflows/<workflow_name>/configs/aws.yaml` — AWS production config
- `steps/monitoring/` — global drift detection, log collection, retrain trigger (shared across all workflows)
- [.github/workflows/ci.yml](.github/workflows/ci.yml) — CI/CD pipeline

**AWS stack components**:
| Component | Name | AWS Service |
|---|---|---|
| Orchestrator | `sagemaker_orch` | SageMaker Pipelines |
| Artifact Store | `s3_store` | S3 (`aips-zenml-artifacts`) |
| Container Registry | `ecr_registry` | ECR |
| Experiment Tracker | `mlflow_tracker` | Self-hosted MLflow on EC2 |

---

### ServingEngineer

**Responsibility**: Batch and real-time recommendation serving.

**Owned steps**: `generate_batch_recommendations`, `build_serving_image`, `deploy_endpoint`

**Common commands**:

```bash
# Run serving pipeline (builds + deploys)
make run-local-serving WORKFLOW=<workflow_name>

# Start FastAPI server locally for testing
make serve-local WORKFLOW=<workflow_name>
# Then test: curl -X POST http://localhost:8080/recommend -H "Content-Type: application/json" -d '{"user_id": 1, "top_k": 10}'

# Build serving Docker image manually
make docker-build-serving WORKFLOW=<workflow_name>

# Health check
curl http://localhost:8080/health
```

**API reference** (`workflows/<workflow_name>/serving/app.py`):

| Endpoint     | Method | Request Body                 | Response                                                                    |
| ------------ | ------ | ---------------------------- | --------------------------------------------------------------------------- |
| `/health`    | GET    | —                            | `{status, model_version, n_users, n_items, rank}`                           |
| `/recommend` | POST   | `{user_id: int, top_k: int}` | `{user_id, recommendations: [{item_id, score}], model_version, latency_ms}` |

**DynamoDB schema** (`movie-recommendations` table):

- Partition key: `userId` (String)
- Attribute: `recommendations` (JSON string: `[{item_id, score}, ...]`)
- TTL: `updated_at` (48h from batch job time)

---

## Creating a New Pipeline

Use the `create-e2e-ml-workflow` agent skill (see [.agents/skills/create-e2e-ml-workflow/SKILL.md](.agents/skills/create-e2e-ml-workflow/SKILL.md)).

**Quick summary**:

1. Copy `workflows/matrix_factorization/` to `workflows/<your_workflow_name>/`
2. Update all imports from `workflows.matrix_factorization.` → `workflows.<your_workflow_name>.`
3. Update `run.py` to include the new workflow pipelines in the `PIPELINES` set and imports
4. Create `workflows/<your_workflow_name>/configs/local.yaml` and `aws.yaml`
5. Add tests in `workflows/<your_workflow_name>/tests/unit/`

---

## Monitoring & Retraining

The monitoring pipeline runs daily (configurable) and checks:

1. **Data drift**: Evidently `DataDriftPreset` comparing recent inference users vs. training data
2. **Model age**: If > `max_age_days` (default 30) since last training

When triggered, `trigger_retraining` fires `training_pipeline` with `enable_cache=False` to ensure fresh retraining.

**Drift thresholds** (in `workflows/<workflow_name>/configs/aws.yaml`):

```yaml
drift_threshold_n_features: 2 # retrain if >2 features drift
max_age_days: 30 # retrain if model is >30 days old
```

**Manual retrain trigger**:

```bash
make run-aws-training WORKFLOW=<workflow_name>
# or: uv run python run.py run --workflow <workflow_name> --pipeline training --config workflows/<workflow_name>/configs/aws.yaml --stack aws_stack --no-cache
```

---

## Running Tests

```bash
# Unit tests only (fast, no ZenML/AWS required)
make test WORKFLOW=<workflow_name>

# Integration tests
make test-integration WORKFLOW=<workflow_name>

# All tests with coverage
make test-all WORKFLOW=<workflow_name>

# Single module (run directly)
uv run pytest workflows/<workflow_name>/tests/unit/test_checkpointing.py -v
```

---

## Key Conventions

1. **Never use `pipeline` as a variable name** — it shadows the ZenML decorator
2. **All ZenML step outputs must be typed and annotated** — required for artifact tracking
3. **Import pipelines/steps from the module, not from `__init__`** — prevents circular imports in `run.py`
4. **Configs in `configs/*.yaml` control all environment differences** — no code changes needed to switch environments
5. **Checkpoint paths in configs** — `./checkpoints` for local, `s3://aips-zenml-checkpoints` for AWS
6. **All S3/local path operations go through `utils/checkpointing.py`** — `s3fs` makes both transparent
7. **Global steps live in `steps/`** — cross-workflow steps (e.g. monitoring) go in `steps/<domain>/`; workflow-specific steps go in `workflows/<workflow_name>/steps/`. Import global steps with `from steps.<domain>.<module> import <step>`.
