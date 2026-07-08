---
name: create-e2e-ml-workflow
description: Creates a new end-to-end ZenML ML workflow from scratch.
---

# Create a New ZenML ML Workflow

## Overview

Set up a new end-to-end ZenML ML workflow patterned after the production-ready `workflows/matrix_factorization` implementation.

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

1. **Workflow name** — snake_case identifier, e.g. `content_based_filtering`
2. **Model class name** — PascalCase, e.g. `ContentFilterModel`
3. **Model ZenML name** — the registered model name, e.g. `content_filter_recommender`
4. **Algorithm** — what training algorithm is used (determines the training step internals)
5. **Data source** — where training data comes from (download URL, S3 path, database, etc.)
6. **Serving mode** — real-time API only / batch only / both

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

### `steps/data_ingestion/ingest.py`

> **Stub:** [`stubs/steps/data_ingestion/ingest.py`](stubs/steps/data_ingestion/ingest.py.stub) — adapt loader/parsing logic to your dataset while preserving typed output + Dask materializer usage.

### `steps/data_validation/validate.py`

> **Stub:** [`stubs/steps/data_validation/validate.py`](stubs/steps/data_validation/validate.py.stub) — adjust required columns and thresholds for your workflow.

### `steps/feature_engineering/encoders.py`

> **Stub:** [`stubs/steps/feature_engineering/encoders.py`](stubs/steps/feature_engineering/encoders.py.stub) — update ID column names as needed.

### `steps/feature_engineering/split.py`

> **Stub:** [`stubs/steps/feature_engineering/split.py`](stubs/steps/feature_engineering/split.py.stub) — keep per-entity temporal split pattern to avoid leakage.

### `steps/hpo/run_hpo.py`

> **Stub:** [`stubs/steps/hpo/run_hpo.py`](stubs/steps/hpo/run_hpo.py.stub) — preserve resumable Optuna study + distributed one-trial-per-future pattern.

### `steps/training/train.py` (or `train_<algo>.py`)

The most important step. Implement with the full checkpointing protocol.

> **Stub:** [`stubs/steps/training/train.py`](stubs/steps/training/train.py.stub) — preserve epoch-level checkpoint + resume behavior.

### `steps/model_evaluation/evaluate.py`

> **Stub:** [`stubs/steps/model_evaluation/evaluate.py`](stubs/steps/model_evaluation/evaluate.py.stub) — keep ranking + regression metric structure; adapt metrics to task.

### `steps/model_evaluation/register.py`

> **Stub:** [`stubs/steps/model_evaluation/register.py`](stubs/steps/model_evaluation/register.py.stub) — keep metadata logging + quality gate + checkpoint cleanup.

### Serving steps

- `steps/serving/batch_predict.py` → [`stubs/steps/serving/batch_predict.py`](stubs/steps/serving/batch_predict.py.stub)
- `steps/serving/build_image.py` → [`stubs/steps/serving/build_image.py`](stubs/steps/serving/build_image.py.stub)
- `steps/serving/deploy.py` → [`stubs/steps/serving/deploy.py`](stubs/steps/serving/deploy.py.stub)

These match the production matrix-factorization serving flow.

---

## Step 7: Create Pipelines

### `pipelines/__init__.py`

> **Stub:** [`stubs/pipelines/__init__.py`](stubs/pipelines/__init__.py.stub) — replace `<workflow_name>`.

### `pipelines/training_pipeline.py`

> **Stub:** [`stubs/pipelines/training_pipeline.py`](stubs/pipelines/training_pipeline.py.stub) — replace `<workflow_name>` and `<model_zenml_name>`. Keep data ingestion/validation/feature engineering and optional HPO inside this pipeline before training.

### `pipelines/serving_pipeline.py`

> **Stub:** [`stubs/pipelines/serving_pipeline.py`](stubs/pipelines/serving_pipeline.py.stub) — replace `<workflow_name>`.

### `pipelines/monitoring_pipeline.py`

> **Stub:** [`stubs/pipelines/monitoring_pipeline.py`](stubs/pipelines/monitoring_pipeline.py.stub) — replace `<workflow_name>`. Keep monitoring self-contained by ingesting the reference dataset inside the pipeline (do not require `raw_ratings` as a pipeline input).

---

## Step 8: Create the FastAPI Serving App

### `serving/app.py`

> **Stub:** [`stubs/serving/app.py`](stubs/serving/app.py.stub) — replace `<workflow_name>`, `<ModelClassName>`, and `<WorkflowName>`.

### `serving/Dockerfile`

> **Stub:** [`stubs/serving/Dockerfile`](stubs/serving/Dockerfile.stub) — replace `<workflow_name>`.

---

## Step 9: Create the ZenML Step Dockerfile

### `workflows/<workflow_name>/Dockerfile`

Built from the repo root: `docker build -f workflows/<workflow_name>/Dockerfile .`

> **Stub:** [`stubs/Dockerfile`](stubs/Dockerfile.stub) — copy verbatim (no placeholders).

---

## Step 10: Add Tests

### `tests/unit/test_**.py`

Add unit tests for critical workflow-specific logic first:

- model predict/batch_predict behavior
- algorithm utility kernels
- serving API happy-path + error-path

Use `pytest` + mocking for external systems (S3, Dask scheduler, SageMaker, DynamoDB).

> **Stub:** [`stubs/tests/unit/test_workflow_model.py`](stubs/tests/unit/test_workflow_model.py.stub)

---

## Step 11: Update Shared Documentation

**`README.md`** — add rows to the Pipeline Reference table:

```markdown
| `training` | `make run-local-training WORKFLOW=<workflow_name>` | Ingest → validate → encode → split → optional HPO → train → evaluate → register |
| `serving` | `make run-local-serving WORKFLOW=<workflow_name>` | Batch recs + real-time endpoint |
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
