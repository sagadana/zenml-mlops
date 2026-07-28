# Workflow: Matrix Factorization

## Confirmed Decisions

| Decision                | Choice                                                                               | Rationale                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| **Algorithm**           | **ALS** (not SVD)                                                                    | Handles implicit feedback and supports BLAS-backed training via `implicit` |
| **ZenML Server**        | Local compose stack (dev) and remote AWS stack (prod)                                | Shared metadata store + dashboard across environments                      |
| **Serving**             | Both batch (S3 + optional DynamoDB) and real-time (FastAPI + local/SageMaker deploy) | Batch for pre-computation; real-time for low-latency fallback              |
| **Dataset**             | MovieLens 1M (local) / MovieLens 25M (AWS)                                           | Controlled by `dataset_size` pipeline parameter                            |
| **Monitoring**          | Evidently AI                                                                         | Purpose-built ML monitoring with ZenML-compatible workflow                 |
| **Experiment tracking** | ZenML native (log_metadata)                                                          | Built-in metadata logging without external dependency                      |
| **Checkpointing**       | Epoch-level `.npy` + `.done` marker files                                            | Resumable training with atomic checkpoint commits                          |
| **ALS backend**         | `implicit.als.AlternatingLeastSquares` (default)                                     | BLAS-backed, GPU-optional; handles all parallelism internally              |
| **Pluggable model**     | `recommender_class_name` config param (any `BaseRecommender`)                        | Swap ALS backends or algorithms without touching pipeline code             |

---

## Architecture Overview

```mermaid
graph TD

    subgraph D[data_pipeline]
        D1[ingest_data] --> D2[validate_data] --> D2a[preprocess_data] --> D3[build_encoders] --> D4[create_features_artifact]
    end

    subgraph T[training_pipeline]
        T0[load_features_artifact] --> T1[prepare_features]
        T1 --> T5[run_hpo_trial xN optional]
        T1 --> T4[split_data xHPO only]
        T4 --> T5
        T5 --> T6[collect_best_hpo_params]
        T1 --> T7[train_als full dataset]
        T6 --> T7
        T7 --> T8[visualize_training]
        T7 --> T11[register_model]
    end

    subgraph BI[batch_inference_pipeline]
        S1[load_als_model] --> S1a["get_total_users (→ total_users: int, batch_size: int)"]
        S1a --> S2[predict_user_batch xN] --> S3[collect_batch_inference_report]
    end

    subgraph DP[deployment_pipeline]
        S4[get_model_artifact_uri] --> S5[build_serving_image] --> S6[deploy_endpoint] --> S6-a(push inference logs)
    end

    subgraph M[monitoring_pipeline]
        M1[load_raw_ratings_artifact] --> M1a[select_reference_features]
        M2[ingest_data] --> M2a[select_comparison_features] --> M3[evidently_report]
        M1a --> M3
        M3 --> M5[check_retrain]
    end

    subgraph OE[online_evaluation_pipeline]
        OE1[load_scaled_ratings_artifact] --> OE1a[select_reference_features]
        OE2[ingest_logs] --> OE2a[select_current_features] --> OE3[evidently_report]
        OE1a --> OE3
    end

    A[MovieLens Dataset] --> D
    D -->|"trigger(TBC)"| T
    T -->|"trigger(TBC)"| BI
    T -->|"trigger(TBC)"| DP
    DP -->|"schedule(TBC)"| M
    DP -->|"schedule(TBC)"| OE
    S6-a -->|logs| OE2
    M -->|"trigger(TBC)"| D

```

_TBC: Means "to be confirmed" — the exact trigger/scheduling mechanism is not yet finalized due to the limitations in the community version of Zenml, but the intent is to have a fully automated workflow._

---

## Source Layout (Current)

- `workflows/matrix_factorization/configs/local/{data_pipeline,training_pipeline,batch_inference_pipeline,deployment_pipeline,monitoring_pipeline,online_evaluation_pipeline}.yaml`
- `workflows/matrix_factorization/configs/aws/{data_pipeline,training_pipeline,batch_inference_pipeline,deployment_pipeline,monitoring_pipeline,online_evaluation_pipeline}.yaml`
- `workflows/matrix_factorization/materializers/als_recommender_materializer.py`
- `workflows/matrix_factorization/models/{als_implicit_recommender,als_numba_recommender,base_recommender,numba}.py`
- `workflows/matrix_factorization/pipelines/{data,training,batch_inference,deployment,monitoring,online_evaluation}_pipeline.py`
- `workflows/matrix_factorization/steps/`
  - `data/ingest.py`, `data/validate.py`, `data/preprocess.py`
  - `features/{encoders,artifacts,select,split}.py` (`split.py` exports `prepare_features` + `split_data`)
  - `hpo/run_hpo.py` (`run_hpo_trial`, `collect_best_hpo_params`)
  - `training/train_als.py` (`train_als` — full-dataset training loop with inline checkpoint resume and optional warm start)
  - `evaluation/{evaluate,register}.py`
  - `prediction/{batch_predict,batch_predict_user}.py`
- `workflows/matrix_factorization/serving/app.py`
- shared helpers: `helpers/checkpointing.py`, `helpers/resource_monitor.py` (per-epoch CPU/memory/GPU snapshots), `helpers/pipeline.py` (pipeline trigger + discovery)
- shared retrain/trigger helpers: `steps/retrain.py`, `steps/trigger.py`
- shared serving steps: `steps/serving/{build_image,deploy_model,model_artifacts}.py`

---

## Pipeline Definitions

### Training pipeline (`training_pipeline`)

Order:

1. `load_features_artifact`
2. `prepare_features` (applies encoders to full dataset; always run before training)
3. `split_data` (only within HPO path)
4. `run_hpo_trial` (fan-out, optional via `enable_hpo`)
5. `collect_best_hpo_params` (fan-in, optional via `enable_hpo`)
6. `train_als` (trains on full `features` from step 2 with inline checkpoint resume; supports warm start from a previous model stage)
7. `visualize_training`
8. `register_model` (metrics sourced from `training_states` — no separate eval step)

### Data pipeline (`data_pipeline`)

Order:

1. `ingest_data`
2. `validate_data`
3. `preprocess_data` (dedup, user/item activity filter, top-N per user)
4. `build_encoders`
5. `create_features_artifact`

Bound ZenML model: `als_movie_recommender`.

### Batch inference pipeline (`batch_inference_pipeline`)

Runs batch scoring fan-out/fan-in:

- `load_als_model` → `get_total_users` (→ `total_users: int`, `batch_size: int`) → `predict_user_batch` × n_batches (fan-out, each computes its own slice) → `collect_batch_inference_report` (fan-in)

Outputs to S3 parquet and optionally DynamoDB.

### Deployment pipeline (`deployment_pipeline`)

Builds and deploys the real-time serving endpoint:

- `get_model_artifact_uri` → `build_serving_image` → `deploy_endpoint`

### Monitoring pipeline (`monitoring_pipeline`)

Order:

1. `load_raw_ratings_artifact` → `select_feature_columns(id="select_reference_features")` (training baseline)
2. `ingest_data` → `select_feature_columns(id="select_comparison_features")` (new/recent data)
3. `evidently_report` (single report, `DataQualityPreset` + `DataDriftPreset`)
4. `check_retrain` (evaluates the report; emits `should_retrain`)

Retrain target:

- module: `workflows.matrix_factorization.pipelines.training_pipeline`
- function: `training_pipeline`

### Online evaluation pipeline (`online_evaluation_pipeline`)

Order:

1. `load_scaled_ratings_artifact` → `select_feature_columns(id="select_reference_features")` (ground-truth training ratings)
2. `ingest_logs` → `select_feature_columns(id="select_current_features")` (recent model predictions)
3. `evidently_report` (PrecisionTopK, RecallTopK, NDCG, MAP, ScoreDistribution at k=10)

Observability only — no retrain trigger.

---

## Configuration Contract

### `configs/local/training_pipeline.yaml`

Core values:

- `dataset_size: "1m"`
- `enable_hpo: true`
- `optuna_storage: "${OPTUNA_STORAGE_URI}"`
- `checkpoint_path: "s3://${ZENML_CHECKPOINT_BUCKET}"`
- `settings.docker.dockerfile: "docker/pipeline/Dockerfile"`

### `configs/local/data_pipeline.yaml`

Core values:

- `dataset_size: "1m"`
- validation thresholds for sparse ratings data
- `create_features_artifact` persists encoder artifact

### `configs/local/batch_inference_pipeline.yaml`

Core values:

- `n_batches: 3`
- `model_stage: "staging"`
- `batch_output_path: "s3://${ZENML_PREDICTIONS_BUCKET}/batch"`

### `configs/local/deployment_pipeline.yaml`

Core values:

- `deploy_mode: "local"`
- `endpoint_name: "als-movie-recommender"`

### `configs/local/monitoring_pipeline.yaml`

Core values:

- `logs_path: "s3://${ZENML_PREDICTIONS_BUCKET}/logs"`
- `monitoring_output_path: "s3://${ZENML_PREDICTIONS_BUCKET}/monitoring"`
- `retrain_config_path: "workflows/matrix_factorization/configs/local/training_pipeline.yaml"`

### `configs/local/online_evaluation_pipeline.yaml`

Core values:

- `ingest_logs.runtime: inline`
- `logs_path: "s3://${ZENML_PREDICTIONS_BUCKET}/logs"`
- `lookback_days: 30`

### `configs/aws/training_pipeline.yaml`

Core values:

- `dataset_size: "25m"`
- `enable_hpo: true`
- `optuna_storage: "${OPTUNA_STORAGE_URI}"`
- `checkpoint_path: "s3://zenml-checkpoints"`

### `configs/aws/data_pipeline.yaml`

Core values:

- `dataset_size: "25m"`
- validation thresholds for sparse ratings data
- `create_features_artifact` persists encoder artifact

### `configs/aws/batch_inference_pipeline.yaml`

Core values:

- `n_batches: 17`
- `batch_output_path: "s3://zenml-predictions/batch"`
- `dynamodb_table: "${ZENML_BATCH_DDB_TABLE_NAME}"`

### `configs/aws/deployment_pipeline.yaml`

Core values:

- `deploy_mode: "sagemaker"`
- `instance_type: "ml.t2.medium"`

### `configs/aws/monitoring_pipeline.yaml`

Core values:

- `monitoring_output_path: "s3://zenml-predictions/monitoring"`
- `retrain_config_path: "workflows/matrix_factorization/configs/aws/training_pipeline.yaml"`
- `settings.docker.dockerfile: "docker/pipeline/Dockerfile"`

### `configs/aws/online_evaluation_pipeline.yaml`

Core values:

- `ingest_logs.runtime: inline`
- `logs_path: "s3://zenml-predictions/logs"`
- `lookback_days: 30`

---

## Serving API Contract (`serving/app.py`)

Endpoints:

- `GET /health` -> `{status, app_version, model_version, n_users, n_items, factors, cpu_percent, memory_percent, disk_percent}`
- `POST /predict` with `{user_id, top_k}` -> `{user_id, predictions, model_version, latency_ms}`

Behavior:

- Resolves `MODEL_URI` to `MODEL_PATH` on startup; `MODEL_URI` may be local or `s3://...`.
- Writes inference logs to `MODEL_INFERENCE_LOG_PATH` when `MODEL_INFERENCE_LOG_ENABLED=true`.
- Returns 404 for unknown users.

---

## AWS Stack Components (from `infra/aws/setup_stacks.sh`)

| Component          | Name                       |
| ------------------ | -------------------------- |
| Service Connector  | `aws_connector`            |
| Artifact Store     | `s3_store`                 |
| Container Registry | `ecr_registry`             |
| Orchestrator       | `sagemaker_orchestrator`   |
| Step Operator      | `sagemaker_step_operator`  |
| Image Builder      | `local_image_builder`      |
| Data Validator     | `evidently_data_validator` |
| Stack              | `aws_stack`                |

---
