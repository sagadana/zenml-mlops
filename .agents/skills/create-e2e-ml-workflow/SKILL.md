---
name: create-e2e-ml-workflow
description: Creates a new end-to-end ZenML ML workflow from scratch.
updated_at: 2026-07-23T00:00:00Z
---

# Create a New ZenML ML Workflow

## Overview

Set up a new end-to-end ZenML ML workflow patterned after the production-ready workflow template in this repository.

Every workflow lives under `workflows/<workflow_name>/` and is self-contained for:

- configs
- models
- materializers
- steps (data, features, hpo, training, evaluation, prediction)
- pipelines
- serving app

## When to Use

- Adding a new ML use case (content-based filtering, click-through prediction, ranking, etc.)
- Any time a new model or training loop needs the full pipeline: data → train → serve → monitor

> **Stubs:** All templates live in [`stubs/`](stubs/). Before writing files, read the stub content, then replace placeholders:
>
> - `<service_name>`: the name of the service
> - `<workflow_name>`: snake_case workflow directory/module name
> - `<model_name>`: model name
> - `<ModelClassName>`: PascalCase model class
> - `<WorkflowName>`: PascalCase workflow name

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

Use pipeline-level `parameters:` only for true pipeline controls. Put step inputs in `steps.<step_name>.parameters` and avoid passing them through pipeline function signatures.

### `workflows/<workflow_name>/configs/__init__.py`

> **Stub:** [`stubs/configs/__init__.py`](stubs/configs/__init__.py.stub) — replace `<workflow_name>` and `<ModelClassName>`.

### `workflows/<workflow_name>/configs/local/training_pipeline.yaml`

> **Stub:** [`stubs/configs/local/training_pipeline.yaml`](stubs/configs/local/training_pipeline.yaml.stub) — replace `<workflow_name>` placeholders.

### `workflows/<workflow_name>/configs/local/batch_inference_pipeline.yaml`

> **Stub:** [`stubs/configs/local/batch_inference_pipeline.yaml`](stubs/configs/local/batch_inference_pipeline.yaml.stub) — replace `<workflow_name>` placeholders.

### `workflows/<workflow_name>/configs/local/deployment_pipeline.yaml`

> **Stub:** [`stubs/configs/local/deployment_pipeline.yaml`](stubs/configs/local/deployment_pipeline.yaml.stub) — replace `<workflow_name>` and `<service_name>` placeholders.

### `workflows/<workflow_name>/configs/local/monitoring_pipeline.yaml`

> **Stub:** [`stubs/configs/local/monitoring_pipeline.yaml`](stubs/configs/local/monitoring_pipeline.yaml.stub) — replace `<workflow_name>` placeholders.

### `workflows/<workflow_name>/configs/local/online_evaluation_pipeline.yaml`

> **Stub:** [`stubs/configs/local/online_evaluation_pipeline.yaml`](stubs/configs/local/online_evaluation_pipeline.yaml.stub) — replace `<workflow_name>` placeholders.

### `workflows/<workflow_name>/configs/local/data_pipeline.yaml`

> **Stub:** [`stubs/configs/local/data_pipeline.yaml`](stubs/configs/local/data_pipeline.yaml.stub) — replace `<workflow_name>` placeholders.

### `workflows/<workflow_name>/configs/aws/training_pipeline.yaml`

> **Stub:** [`stubs/configs/aws/training_pipeline.yaml`](stubs/configs/aws/training_pipeline.yaml.stub) — replace `<workflow_name>` and configure env var overrides for PostgreSQL storage and S3 paths.

### `workflows/<workflow_name>/configs/aws/batch_inference_pipeline.yaml`

> **Stub:** [`stubs/configs/aws/batch_inference_pipeline.yaml`](stubs/configs/aws/batch_inference_pipeline.yaml.stub) — replace `<workflow_name>` and configure DynamoDB/S3 paths for production.

### `workflows/<workflow_name>/configs/aws/deployment_pipeline.yaml`

> **Stub:** [`stubs/configs/aws/deployment_pipeline.yaml`](stubs/configs/aws/deployment_pipeline.yaml.stub) — replace `<workflow_name>` placeholders.

### `workflows/<workflow_name>/configs/aws/monitoring_pipeline.yaml`

> **Stub:** [`stubs/configs/aws/monitoring_pipeline.yaml`](stubs/configs/aws/monitoring_pipeline.yaml.stub) — replace `<workflow_name>` placeholders.

### `workflows/<workflow_name>/configs/aws/online_evaluation_pipeline.yaml`

> **Stub:** [`stubs/configs/aws/online_evaluation_pipeline.yaml`](stubs/configs/aws/online_evaluation_pipeline.yaml.stub) — replace `<workflow_name>` placeholders.

### `workflows/<workflow_name>/configs/aws/data_pipeline.yaml`

> **Stub:** [`stubs/configs/aws/data_pipeline.yaml`](stubs/configs/aws/data_pipeline.yaml.stub) — replace `<workflow_name>` placeholders.

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

### `workflows/<workflow_name>/materializers/__init__.py`

> **Stub:** [`stubs/materializers/__init__.py`](stubs/materializers/__init__.py.stub) — replace `<workflow_name>` and `<ModelClassName>`.

---

## Step 5: Use Shared Utilities

### Shared helpers (no copy needed)

`checkpointing` lives in the root `helpers/` package and is **shared across all workflows**. Do not copy or recreate it — just import directly:

```python
from helpers.checkpointing import save_checkpoint, load_latest_checkpoint, clean_run_checkpoints, list_checkpoints
```

The current reference workflow has no required per-workflow `utils/` package. Add one only when a new workflow has reusable algorithm helpers that do not belong in models or steps.

---

## Step 6: Create Steps

### ZenML step conventions (apply to every step)

- Decorated with `@step(enable_cache=True)` — use `False` for non-deterministic steps
- Every output typed with `Annotated[Type, "artifact_name"]`
- Logging via `logging.getLogger(__name__)` — captured to ZenML dashboard automatically
- Heavy third-party imports (`mlflow`, `optuna`, `evidently`) go inside the function body
- No global state; steps are pure functions of their inputs

### `workflows/<workflow_name>/steps/data/ingest.py`

> **Stub:** [`stubs/steps/data/ingest.py`](stubs/steps/data/ingest.py.stub) — adapt loader/parsing logic to your dataset while preserving typed pandas output.

### `workflows/<workflow_name>/steps/data/validate.py`

> **Stub:** [`stubs/steps/data/validate.py`](stubs/steps/data/validate.py.stub) — adjust required columns and thresholds for your workflow.

### `workflows/<workflow_name>/steps/data/preprocess.py`

> **Stub:** [`stubs/steps/data/preprocess.py`](stubs/steps/data/preprocess.py.stub) — adjust filtering thresholds and top-N logic for your dataset.

### `workflows/<workflow_name>/steps/features/encoders.py`

> **Stub:** [`stubs/steps/features/encoders.py`](stubs/steps/features/encoders.py.stub) — update ID column names as needed.

### `workflows/<workflow_name>/steps/features/split.py`

> **Stub:** [`stubs/steps/features/split.py`](stubs/steps/features/split.py.stub) — keep per-entity temporal split pattern to avoid leakage.

### `workflows/<workflow_name>/steps/features/artifacts.py`

> **Stub:** [`stubs/steps/features/artifacts.py`](stubs/steps/features/artifacts.py.stub) — persist encoders in `data_pipeline` and load them in `training_pipeline` by artifact name.

### `workflows/<workflow_name>/steps/features/select.py`

> **Stub:** [`stubs/steps/features/select.py`](stubs/steps/features/select.py.stub) — shared column-selection step for monitoring and online evaluation pipelines.

### `workflows/<workflow_name>/steps/hpo/run_hpo.py`

> **Stub:** [`stubs/steps/hpo/run_hpo.py`](stubs/steps/hpo/run_hpo.py.stub) — preserve `run_hpo_trial` fan-out + `collect_best_hpo_params` fan-in pattern with resumable Optuna storage.

### `workflows/<workflow_name>/steps/training/train_<algo>.py`

All epochs are trained in a single step with automatic checkpoint resume. Checkpointing is handled inline (no separate checkpoint steps).

> **Stub:** [`stubs/steps/training/train.py`](stubs/steps/training/train.py.stub) — rename to `train_<algo>.py`; preserve single-step full-training-loop shape with inline checkpoint save/resume.

### `workflows/<workflow_name>/steps/training/visualize.py`

> **Stub:** [`stubs/steps/training/visualize.py`](stubs/steps/training/visualize.py.stub) — render per-epoch training metrics as a ZenML HTML artifact.

### `workflows/<workflow_name>/steps/evaluation/evaluate.py`

> **Stub:** [`stubs/steps/evaluation/evaluate.py`](stubs/steps/evaluation/evaluate.py.stub) — keep evaluation logic task-aware; select metrics appropriate for your ML task (classification/regression/ranking/forecasting).

### `workflows/<workflow_name>/steps/evaluation/register.py`

> **Stub:** [`stubs/steps/evaluation/register.py`](stubs/steps/evaluation/register.py.stub) — keep metadata logging + quality gate + checkpoint cleanup.

### Prediction steps

- `workflows/<workflow_name>/steps/prediction/batch_predict.py` → [`stubs/steps/prediction/batch_predict.py`](stubs/steps/prediction/batch_predict.py.stub)
- `workflows/<workflow_name>/steps/prediction/batch_predict_user.py` → [`stubs/steps/prediction/batch_predict_user.py`](stubs/steps/prediction/batch_predict_user.py.stub)

`build_serving_image` and `deploy_endpoint` are shared global steps under `steps/serving/` (not per-workflow files).

---

## Step 7: Create Pipelines

### `pipelines/__init__.py`

> **Stub:** [`stubs/pipelines/__init__.py`](stubs/pipelines/__init__.py.stub) — replace `<workflow_name>`.

### `pipelines/training_pipeline.py`

> **Stub:** [`stubs/pipelines/training_pipeline.py`](stubs/pipelines/training_pipeline.py.stub) — replace `<workflow_name>` and `<model_name>`. Keep training focused on `load_features_artifact + ingest_data -> split_data -> HPO/train/eval/register`.

### `pipelines/data_pipeline.py`

> **Stub:** [`stubs/pipelines/data_pipeline.py`](stubs/pipelines/data_pipeline.py.stub) — replace `<workflow_name>`. Keep `ingest_data -> validate_data -> preprocess_data -> build_encoders -> create_features_artifact` in this pipeline.

### `pipelines/batch_inference_pipeline.py`

> **Stub:** [`stubs/pipelines/batch_inference_pipeline.py`](stubs/pipelines/batch_inference_pipeline.py.stub) — replace `<workflow_name>`. Preserves fan-out `get_user_batch_slice × n_batches` + `predict_user_batch × n_batches` + fan-in `collect_batch_recommendations`. `get_user_ids` computes both the full id list and effective `batch_size` from `n_batches` and `min_user_batch_size`.

### `pipelines/deployment_pipeline.py`

> **Stub:** [`stubs/pipelines/deployment_pipeline.py`](stubs/pipelines/deployment_pipeline.py.stub) — replace `<workflow_name>`. Preserves `get_model_artifact_uri → build_serving_image → deploy_endpoint` flow.

### `pipelines/monitoring_pipeline.py`

> **Stub:** [`stubs/pipelines/monitoring_pipeline.py`](stubs/pipelines/monitoring_pipeline.py.stub) — replace `<workflow_name>`. Compares freshly ingested data against the stored training baseline using Evidently DataQualityPreset + DataDriftPreset.  `ingest_data(lookback_days)` provides the reference (new data); `load_raw_ratings_artifact` provides the comparison (training baseline). `check_retrain_trigger` evaluates the report and emits `should_retrain`.

### `pipelines/online_evaluation_pipeline.py`

> **Stub:** [`stubs/pipelines/online_evaluation_pipeline.py`](stubs/pipelines/online_evaluation_pipeline.py.stub) — replace `<workflow_name>`. Evaluates online ranking quality using Evidently Ranking metrics (PrecisionTopK, RecallTopK, NDCG, MAP, ScoreDistribution at k=10). `load_scaled_ratings_artifact` is the ground-truth reference; `ingest_logs` is the current predictions dataset.

---

## Step 8: Create the FastAPI Serving App

### `serving/app.py`

> **Stub:** [`stubs/serving/app.py`](stubs/serving/app.py.stub) — replace `<workflow_name>`, `<ModelClassName>`, and `<WorkflowName>`. The stub includes `psutil` for system metrics (`cpu_percent`, `memory_percent`, `disk_percent`) in the `/health` response — keep this for operational observability.

### `serving/__init__.py`

> **Stub:** [`stubs/serving/__init__.py`](stubs/serving/__init__.py.stub) — empty init file; required so the serving app module can be imported. `setup.sh` creates it automatically with `touch`.

### Serving Dockerfile

No per-workflow serving Dockerfile is needed. All workflows share `docker/serving/Dockerfile` at the repo root.
It is parameterised via `ARG WORKFLOW` — `build_serving_image` passes `--build-arg WORKFLOW=<workflow_name>` automatically.

---

## Step 9: Pipeline Steps Dockerfile

No per-workflow pipeline Dockerfile is needed. All workflows share `docker/pipeline/Dockerfile` at the repo root.
Reference it in `configs/local/training_pipeline.yaml` and `configs/aws/training_pipeline.yaml`:

```yaml
settings:
  docker:
    dockerfile: "docker/pipeline/Dockerfile"
```

---

## Step 10: Update Shared Documentation

**`README.md`** — add rows to the Pipeline Reference table:

```markdown
| `training` | `make run-local-training WORKFLOW=<workflow_name>` | Ingest → validate → encode → split → optional HPO → train → evaluate → register |
| `data` | `make run-local-pipeline WORKFLOW=<workflow_name> PIPELINE=data_pipeline` | Ingest → validate → preprocess → encode → save features artifact |
| `batch-inference` | `make run-local-batch-inference WORKFLOW=<workflow_name>` | Batch recs fan-out/fan-in → S3 + DynamoDB |
| `deployment` | `make run-local-deployment WORKFLOW=<workflow_name>` | Build serving image → deploy real-time endpoint |
```

**`run.py`** — no registration needed; workflows are auto-discovered at runtime.

> **Stub:** [`stubs/run_additions.py`](stubs/run_additions.py.stub) — read-only reference showing how auto-discovery works and how to verify a new workflow is found. Do not copy this file; it is documentation only.

Verify the new workflow is discoverable:

```bash
python run.py list-workflows
python run.py list-pipelines --workflow <workflow_name>
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

| Library           | Purpose in this project                                                                       | Docs                                   |
| ----------------- | --------------------------------------------------------------------------------------------- | -------------------------------------- |
| **ZenML**         | Orchestration, artifact tracking, model registry                                              | https://docs.zenml.io/                 |
| **Numba**         | JIT-compiled (`@njit`) Mathematical operations; `parallel=True, nogil=True, cache=True` flags | https://numba.readthedocs.io/          |
| **Optuna**        | HPO — `TPESampler`, `HyperbandPruner`, resumable studies via `load_if_exists=True`            | https://optuna.readthedocs.io/         |
| **Evidently AI**  | Drift detection — `DataDriftPreset`, `DataQualityPreset`                                      | https://docs.evidentlyai.com/          |
| **FastAPI**       | Real-time serving app (`/health` + task endpoint such as `/predict`)                          | https://fastapi.tiangolo.com/          |
| **MLflow**        | Experiment tracking; ZenML native integration                                                 | https://mlflow.org/docs/latest/        |
| **NumPy**         | Core numerical arrays/tensors for preprocessing and model logic                               | https://numpy.org/doc/stable/          |
| **Pandas**        | Tabular preprocessing and feature engineering                                                 | https://pandas.pydata.org/docs/        |
| **s3fs**          | Transparent S3 ↔ local filesystem for checkpointing (`np.save`/`np.load` work on both)        | https://s3fs.readthedocs.io/           |
| **psutil**        | System metrics (`cpu_percent`, `memory_percent`, `disk_percent`) in `/health` response        | https://psutil.readthedocs.io/         |
| **scikit-learn**  | Baselines, preprocessing utilities, and evaluation helpers                                    | https://scikit-learn.org/stable/       |
| **AWS SageMaker** | Orchestrator, step operator, endpoint deployment                                              | https://docs.aws.amazon.com/sagemaker/ |
| **AWS DynamoDB**  | Optional low-latency online store for batch inference outputs and lookup features             | https://docs.aws.amazon.com/dynamodb/  |
| **uv**            | Dependency management (`uv run`, `uv add`)                                                    | https://docs.astral.sh/uv/             |

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
10. **Training step function name must match the YAML step key** — if the function is `train_<algo>`, the YAML block must use the same key (not a stale default). Update both `local/training_pipeline.yaml` and `aws/training_pipeline.yaml`.
11. **`serving/__init__.py` must exist** — `setup.sh` creates it. Without it, the serving app module cannot be imported.
12. **Large batch inference must be chunked** — never score records one-by-one for large datasets; use vectorized/chunked prediction (e.g. `batch_size=10_000`) to avoid OOM and throughput collapse.
13. **Avoid pass-through pipeline parameters** — if a value is consumed by a step, define it in that step's YAML config block and call the step without forwarding duplicate pipeline args.
14. **Build features before training** — run `data_pipeline` first so `training_pipeline` can load the named features artifact.
15. **Monitoring/evaluation column selection is a workflow step** — include `steps/features/select.py` whenever monitoring or online evaluation pipelines import `select_feature_columns`.
