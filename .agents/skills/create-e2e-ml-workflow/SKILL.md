---
name: create-e2e-ml-workflow
description: Creates a new end-to-end ZenML ML workflow from scratch.
---

# Create a New ZenML ML Workflow

## Overview

Set up a new end-to-end ZenML ML workflow patterned after the production-ready workflow template in this repository.

Every workflow lives under `workflows/<workflow_name>/` and is self-contained for:

- configs
- models
- materializers
- steps (data, hpo, training, evaluation, serving)
- pipelines
- serving app
- tests

## When to Use

- Adding a new ML use case (content-based filtering, click-through prediction, ranking, etc.)
- Any time a new model or training loop needs the full pipeline: data → train → serve → monitor

> **Stubs:** All templates live in [`stubs/`](stubs/). Before writing files, read the stub content, then replace placeholders:
>
> - `<workflow_name>`: snake_case workflow directory/module name
> - `<ModelClassName>`: PascalCase model class
> - `<model_zenml_name>`: ZenML model name
> - `<WorkflowName>`: display name

---

## Step 0: Gather Requirements

Ask the user (or infer from context) before starting:

1. **Workflow name** — snake_case identifier, e.g. `churn_prediction`
2. **Model class name** — PascalCase, e.g. `ChurnModel`
3. **Model ZenML name** — the registered model name, e.g. `churn_classifier`
4. **ML task + algorithm family** — classification/regression/ranking/forecasting/etc., and which algorithm(s) will be trained (determines training step internals and HPO search space)
5. **Data source(s)** — where training/validation/inference data comes from (API, object storage, data warehouse, database, streaming topic, files, etc.); if local-dev and production datasets differ, capture both
6. **Feature and label contract** — required input features, target/label definition, and any schema/quality constraints that must fail the pipeline when violated
7. **HPO configuration** — whether hyperparameter optimisation is needed; if yes, define trials, search space, pruner/sampler strategy, and storage backend (`sqlite:///` local, `postgresql://` AWS)
8. **Serving mode** — offline only / batch only / real-time API only / both
9. **Model quality gate** — primary metric(s) and threshold(s) required for promotion (e.g. ROC-AUC, F1, RMSE, MAE), and which model stage is allowed for serving
10. **Monitoring + retraining policy** — metrics/drift checks, alert thresholds, SLA expectations, and retraining triggers (performance drop, drift, max age, schedule)
11. **Other workflow-specific requirements** — any additional constraints or requirements for the workflow

---

## Step 1: Create the Directory Structure

Create directories and `__init__.py` upfront.

> **Script:** [`setup.sh`](./setup.sh) — run with `WF=workflows/<workflow_name> bash .agents/skills/create-e2e-ml-workflow/setup.sh`

---

## Step 2: Create Configs

### `workflows/<workflow_name>/configs/local.yaml`

> **Stub:** [`stubs/configs/local.yaml`](stubs/configs/local.yaml.stub) — replace `<workflow_name>` placeholders.

### `workflows/<workflow_name>/configs/aws.yaml`

> **Stub:** [`stubs/configs/aws.yaml`](stubs/configs/aws.yaml.stub) — replace `<workflow_name>` and configure env var overrides for PostgreSQL storage and S3 paths.

---

## Step 3: Create the Model Class

### `workflows/<workflow_name>/models/<workflow_name>_model.py`

> **Stub:** [`stubs/models/workflow_model.py`](stubs/models/workflow_model.py.stub) — replace placeholders and keep model methods serializable (no runtime-only handles).

### `workflows/<workflow_name>/models/__init__.py`

> **Stub:** [`stubs/models/__init__.py`](stubs/models/__init__.py.stub) — replace `<workflow_name>` and `<ModelClassName>`.

---

## Step 4: Create Materializers

### `workflows/<workflow_name>/materializers/model_materializer.py`

> **Stub:** [`stubs/materializers/model_materializer.py`](stubs/materializers/model_materializer.py.stub) — replace `<workflow_name>` and `<ModelClassName>`.

### `workflows/<workflow_name>/materializers/dask_dataframe_materializer.py`

> **Stub:** [`stubs/materializers/dask_dataframe_materializer.py`](stubs/materializers/dask_dataframe_materializer.py.stub) — copy verbatim (no placeholders to replace).

### `workflows/<workflow_name>/materializers/__init__.py`

> **Stub:** [`stubs/materializers/__init__.py`](stubs/materializers/__init__.py.stub) — replace `<workflow_name>` and `<ModelClassName>`.

---

## Step 5: Create Utilities

### Shared helpers (no copy needed)

`checkpointing` and `dask_cluster` live in the root `helpers/` package and are **shared across all workflows**. Do not copy or recreate them — just import directly:

```python
from helpers.checkpointing import save_checkpoint, load_latest_checkpoint, clean_run_checkpoints, list_checkpoints
from helpers.dask_cluster import get_dask_client, get_client_mode_from_config
```

Create workflow-specific algorithm helpers under:

- `workflows/<workflow_name>/utils/`

---

## Step 6: Create Steps

### ZenML step conventions (apply to every step)

- Decorated with `@step(enable_cache=True)` — use `False` for non-deterministic steps
- Every output typed with `Annotated[Type, "artifact_name"]`
- Logging via `logging.getLogger(__name__)` — captured to ZenML dashboard automatically
- Heavy third-party imports (`mlflow`, `optuna`, `evidently`) go inside the function body
- No global state; steps are pure functions of their inputs

### `workflows/<workflow_name>/steps/data_ingestion/ingest.py`

> **Stub:** [`stubs/steps/data_ingestion/ingest.py`](stubs/steps/data_ingestion/ingest.py.stub) — adapt loader/parsing logic to your dataset while preserving typed output + Dask materializer usage.

### `workflows/<workflow_name>/steps/data_validation/validate.py`

> **Stub:** [`stubs/steps/data_validation/validate.py`](stubs/steps/data_validation/validate.py.stub) — adjust required columns and thresholds for your workflow.

### `workflows/<workflow_name>/steps/feature_engineering/encoders.py`

> **Stub:** [`stubs/steps/feature_engineering/encoders.py`](stubs/steps/feature_engineering/encoders.py.stub) — update ID column names as needed.

### `workflows/<workflow_name>/steps/feature_engineering/split.py`

> **Stub:** [`stubs/steps/feature_engineering/split.py`](stubs/steps/feature_engineering/split.py.stub) — keep per-entity temporal split pattern to avoid leakage.

### `workflows/<workflow_name>/steps/hpo/run_hpo.py`

> **Stub:** [`stubs/steps/hpo/run_hpo.py`](stubs/steps/hpo/run_hpo.py.stub) — preserve resumable Optuna study + distributed one-trial-per-future pattern.

### `workflows/<workflow_name>/steps/training/train.py` (or `train_<algo>.py`)

The most important step. Name the function and file after the algorithm (e.g. `train_xgb`, `train_transformer`) and update the corresponding step key in `configs/local.yaml` and `configs/aws.yaml`. Implement with the full checkpointing protocol.

> **Stub:** [`stubs/steps/training/train.py`](stubs/steps/training/train.py.stub) — preserve epoch-level checkpoint + resume behavior.

### `workflows/<workflow_name>/steps/model_evaluation/evaluate.py`

> **Stub:** [`stubs/steps/model_evaluation/evaluate.py`](stubs/steps/model_evaluation/evaluate.py.stub) — keep evaluation logic task-aware; select metrics appropriate for your ML task (classification/regression/ranking/forecasting).

### `workflows/<workflow_name>/steps/model_evaluation/register.py`

> **Stub:** [`stubs/steps/model_evaluation/register.py`](stubs/steps/model_evaluation/register.py.stub) — keep metadata logging + quality gate + checkpoint cleanup.

### Serving steps

- `workflows/<workflow_name>/steps/serving/batch_predict.py` → [`stubs/steps/serving/batch_predict.py`](stubs/steps/serving/batch_predict.py.stub)
- `workflows/<workflow_name>/steps/serving/build_image.py` → [`stubs/steps/serving/build_image.py`](stubs/steps/serving/build_image.py.stub)
- `workflows/<workflow_name>/steps/serving/deploy.py` → [`stubs/steps/serving/deploy.py`](stubs/steps/serving/deploy.py.stub)

---

## Step 7: Create Pipelines

### `pipelines/__init__.py`

> **Stub:** [`stubs/pipelines/__init__.py`](stubs/pipelines/__init__.py.stub) — replace `<workflow_name>`.

### `pipelines/training_pipeline.py`

> **Stub:** [`stubs/pipelines/training_pipeline.py`](stubs/pipelines/training_pipeline.py.stub) — replace `<workflow_name>` and `<model_zenml_name>`. Keep data ingestion/validation/feature engineering and optional HPO inside this pipeline before training.

### `pipelines/serving_pipeline.py`

> **Stub:** [`stubs/pipelines/serving_pipeline.py`](stubs/pipelines/serving_pipeline.py.stub) — replace `<workflow_name>`.

### `pipelines/monitoring_pipeline.py`

> **Stub:** [`stubs/pipelines/monitoring_pipeline.py`](stubs/pipelines/monitoring_pipeline.py.stub) — replace `<workflow_name>`. Keep monitoring self-contained by preparing its own reference dataset/features inside the pipeline (do not require raw training artifacts as pipeline inputs unless explicitly needed).

---

## Step 8: Create the FastAPI Serving App

### `serving/app.py`

> **Stub:** [`stubs/serving/app.py`](stubs/serving/app.py.stub) — replace `<workflow_name>`, `<ModelClassName>`, and `<WorkflowName>`. The stub includes `psutil` for system metrics (`cpu_percent`, `memory_percent`, `disk_percent`) in the `/health` response — keep this for operational observability.

### Serving Dockerfile

No per-workflow serving Dockerfile is needed. All workflows share `docker/serving/Dockerfile` at the repo root.
It is parameterised via `ARG WORKFLOW` — `build_serving_image` passes `--build-arg WORKFLOW=<workflow_name>` automatically.

---

## Step 9: Pipeline Steps Dockerfile

No per-workflow pipeline Dockerfile is needed. All workflows share `docker/pipeline/Dockerfile` at the repo root.
Reference it in `configs/local.yaml` and `configs/aws.yaml`:

```yaml
settings:
  docker:
    dockerfile: "docker/pipeline/Dockerfile"
```

---

## Step 10: Add Tests

### `tests/unit/test_**.py`

Add unit tests for critical workflow-specific logic first:

- model inference contract (`predict`, optional `batch_predict`) behavior
- algorithm/model utility kernels
- feature/label preprocessing and transformation utilities
- serving API happy-path + error-path (if real-time serving is enabled)

Use `pytest` + mocking for external systems (S3, Dask scheduler, SageMaker, DynamoDB).

> **Stub:** [`stubs/tests/unit/test_workflow_model.py`](stubs/tests/unit/test_workflow_model.py.stub)

---

## Step 11: Update Shared Documentation

**`README.md`** — add rows to the Pipeline Reference table:

```markdown
| `training` | `make run-local-training WORKFLOW=<workflow_name>` | Ingest → validate → encode → split → optional HPO → train → evaluate → register |
| `serving` | `make run-local-serving WORKFLOW=<workflow_name>` | Batch inference + real-time endpoint (as configured) |
```

**`AGENTS.md`** — add a new persona section if the workflow requires different expertise:

```markdown
### <WorkflowName>Engineer

**Responsibility**: ...
**Owned steps**: `ingest_data`, `train_model`, ...
**Common commands**: ...
**Files to know**: ...
```

---

## External Documentation Reference

Key library docs to consult when implementing workflow steps:

| Library | Purpose in this project | Docs |
|---|---|---|
| **ZenML** | Orchestration, artifact tracking, model registry | https://docs.zenml.io/ |
| **Dask** | Distributed DataFrames, parallel step execution, `LocalCluster` / ECS workers | https://docs.dask.org/ |
| **Numba** | JIT-compiled (`@njit`) Mathematical operations; `parallel=True, nogil=True, cache=True` flags | https://numba.readthedocs.io/ |
| **Optuna** | HPO — `TPESampler`, `HyperbandPruner`, resumable studies via `load_if_exists=True` | https://optuna.readthedocs.io/ |
| **Evidently AI** | Drift detection — `DataDriftPreset`, `DataQualityPreset` | https://docs.evidentlyai.com/ |
| **FastAPI** | Real-time serving app (`/health` + task endpoint such as `/predict`) | https://fastapi.tiangolo.com/ |
| **MLflow** | Experiment tracking; ZenML native integration | https://mlflow.org/docs/latest/ |
| **NumPy** | Core numerical arrays/tensors for preprocessing and model logic | https://numpy.org/doc/stable/ |
| **Pandas** | Tabular preprocessing and feature engineering | https://pandas.pydata.org/docs/ |
| **s3fs** | Transparent S3 ↔ local filesystem for checkpointing (`np.save`/`np.load` work on both) | https://s3fs.readthedocs.io/ |
| **psutil** | System metrics (`cpu_percent`, `memory_percent`, `disk_percent`) in `/health` response | https://psutil.readthedocs.io/ |
| **scikit-learn** | Baselines, preprocessing utilities, and evaluation helpers | https://scikit-learn.org/stable/ |
| **AWS SageMaker** | Orchestrator, step operator, endpoint deployment | https://docs.aws.amazon.com/sagemaker/ |
| **AWS DynamoDB** | Optional low-latency online store for batch inference outputs and lookup features | https://docs.aws.amazon.com/dynamodb/ |
| **uv** | Dependency management (`uv run`, `uv add`) | https://docs.astral.sh/uv/ |

---

## Critical Conventions

1. **Never name a variable `pipeline`** — it shadows the `@pipeline` ZenML decorator.
2. **All step return types must be `Annotated[Type, "name"]`** — required for ZenML artifact tracking.
3. **No manual registration in `run.py`** — `run.py` auto-discovers workflows/pipelines.
4. **Checkpoint path scoped to `{base}/{pipeline_run_id}/`** — use `get_step_context().pipeline_run.id`. Prevents parallel runs from overwriting each other.
5. **`.done` marker is always written last** — never skip it. It is the atomicity guarantee that makes resume safe.
6. **Absolute imports from repo root** — `from helpers.checkpointing import ...` and `from workflows.<workflow_name>...`. Relative imports can break ZenML artifact tracking.
7. **`enable_cache=False` for side-effectful steps** — HPO, model registration, serving, monitoring.
8. **Monitoring steps are global** — use `steps/monitoring/*` shared modules from pipelines.
9. **`enable_cache=True` for deterministic data/training/eval steps** unless your workflow explicitly requires otherwise.
10. **Training step function name must match the YAML step key** — if the function is `train_<algo>`, the YAML block must use the same key (not a stale default). Update both `local.yaml` and `aws.yaml`.
11. **`serving/__init__.py` must exist** — `setup.sh` creates it. Without it, the serving app module cannot be imported.
12. **Large batch inference must be chunked** — never score records one-by-one for large datasets; use vectorized/chunked prediction (e.g. `batch_size=10_000`) to avoid OOM and throughput collapse.
