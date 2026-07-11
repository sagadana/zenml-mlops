# AIPS Recommendations — ZenML MLOps Platform

End-to-end MLOps platform built on ZenML. Runs locally or on AWS with a single config switch.

## Architecture

```mermaid
graph LR
    D[Dataset] --> TP[training_pipeline]
    TP --> SP[serving_pipeline]
    TP --> MP[monitoring_pipeline]
    MP -->|drift| TP

    SP --> B[Batch recs → S3 + DynamoDB]
    SP --> R[Real-time API → SageMaker/Docker]
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — Python version and package manager
- Docker (for serving + ZenML remote)
- AWS CLI configured (for AWS runs)

## Quick Start — Local Development

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies and set up ZenML
make setup

# 3. Start local infra services (ZenML, MLflow)
#    & Install dependencies and set up ZenML & Register / Activate local ZenML stack components
make up

# 4. List available workflows and pipelines
make list-workflows
make list-pipelines WORKFLOW=<workflow_name>

# 5. Run pipeline
make run-local-pipeline WORKFLOW=<workflow_name> PIPELINE=<pipeline_name>

```

To stop all local infra services:

```bash
docker compose down
# or
make down
```

## Docker files

All Docker assets live under the root `docker/` folder. Every build uses the repo root as context (`-f docker/<service>/Dockerfile .`):

```text
docker/
  pipeline/Dockerfile        # Shared base image for all ZenML pipeline steps
  serving/Dockerfile         # Shared FastAPI serving image (pass --build-arg WORKFLOW=<name>)
  zenml/Dockerfile
  mlflow/Dockerfile
  mlflow/start.sh
  ops-db/init.sh
docker-compose.yml
```

This structure is designed so each service image can be built and pushed to ECR independently, then mapped to separate ECS services/task definitions later.

## AWS Deployment

```bash
# 1. Set environment variables
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=us-east-1
export ZENML_ARTIFACT_BUCKET=aips-recs-zenml-artifacts
export ZENML_CHECKPOINT_BUCKET=aips-recs-zenml-checkpoints
export ZENML_DATA_BUCKET=aips-recs-zenml-data
export ZENML_PREDICTIONS_BUCKET=aips-recs-zenml-predictions
export ZENML_ECR_REPOSITORY=aips-recs-zenml
export ZENML_BATCH_DDB_TABLE_NAME=movie-recommendations
export ZENML_BATCH_DDB_PARTITION_KEY_NAME=userId
export ZENML_AWS_CONNECTOR_NAME=aws_connector
export ZENML_EXEC_ROLE_NAME=aips-recs-zenml-execution-role
export ZENML_EXEC_ROLE_POLICY_NAME=aips-recs-zenml-execution-policy
export ZENML_SCHEDULER_ROLE_NAME=aips-recs-zenml-scheduler-role
export ZENML_SCHEDULER_ROLE_POLICY_NAME=aips-recs-zenml-scheduler-policy
export ZENML_SAGEMAKER_STEP_OPERATOR_NAME=sagemaker_step_operator
export ZENML_SAGEMAKER_STEP_OPERATOR_INSTANCE_TYPE=ml.m5.xlarge
# Optional: group step-operator jobs in a SageMaker experiment
export ZENML_SAGEMAKER_EXPERIMENT_NAME=aips-recs-zenml
export ZENML_AWS_MODEL_REGISTRY_NAME=mlflow_model_registry
export MLFLOW_TRACKING_URI=http://<your-ec2-ip>:5000
export MLFLOW_TRACKING_USERNAME=<username>
export MLFLOW_TRACKING_PASSWORD=<password>

# 2. Provision AWS infra + register AWS ZenML stack
#    (auto-creates zenml-execution-role using an inline policy generated from env vars)
make infra-aws

# 3. Switch to AWS stack
make stack-aws

# 4. Run full pipeline
make run-aws-training WORKFLOW=<workflow_name>
make run-aws-serving WORKFLOW=<workflow_name>
make run-aws-monitoring WORKFLOW=<workflow_name>
```

## Pipeline Reference

| Pipeline                     | Command                                              | Description                                                                     |
| ---------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------- |
| `<workflow_name>-training`   | `make run-local-training WORKFLOW=<workflow_name>`   | Ingest → validate → encode → split → optional HPO → train → evaluate → register |
| `<workflow_name>-serving`    | `make run-local-serving WORKFLOW=<workflow_name>`    | Batch recs + real-time endpoint                                                 |
| `<workflow_name>-monitoring` | `make run-local-monitoring WORKFLOW=<workflow_name>` | Ingest reference data → drift detection → retrain trigger                       |

## Configuration

All environment differences are controlled by config files — no code changes needed:

| Config Path | Scope | Example Values |
| --- | --- | --- |
| `workflows/<workflow_name>/configs/local/training_pipeline.yaml` | Local training | `dataset_size: "1m"`, `optuna_storage: ${OPS_DB_URI}/...`, `checkpoint_path: "./checkpoints"` |
| `workflows/<workflow_name>/configs/local/serving_pipeline.yaml` | Local serving | `deploy_mode: "local"`, `batch_output_path: "./predictions/batch"` |
| `workflows/<workflow_name>/configs/local/monitoring_pipeline.yaml` | Local monitoring | `logs_path: "./logs/inference"`, `retrain_config_path: .../local/training_pipeline.yaml` |
| `workflows/<workflow_name>/configs/aws/training_pipeline.yaml` | AWS training | `dataset_size: "25m"`, `checkpoint_path: "s3://..."`, `step_operator: true` on compute-heavy steps |
| `workflows/<workflow_name>/configs/aws/serving_pipeline.yaml` | AWS serving | `deploy_mode: "sagemaker"`, `batch_output_path: "s3://${ZENML_PREDICTIONS_BUCKET}/batch"`, `step_operator: true` on batch generation |
| `workflows/<workflow_name>/configs/aws/monitoring_pipeline.yaml` | AWS monitoring | `logs_path: "s3://.../logs"`, `retrain_config_path: .../aws/training_pipeline.yaml`, `step_operator: true` on heavy steps |

## Adding a New Pipeline

See [AGENTS.md](AGENTS.md) and [.agents/skills/create-e2e-ml-workflow/SKILL.md](.agents/skills/create-e2e-ml-workflow/SKILL.md) for the full guide.

## Running Tests

```bash
make test WORKFLOW=<workflow_name>              # unit tests only
make test-integration WORKFLOW=<workflow_name>  # full pipeline integration tests
make test-all WORKFLOW=<workflow_name>          # everything with coverage
```

## Make Commands Reference

All commands are grouped to mirror the Makefile sections.

### Environment (.env)

| Command | Description |
| --- | --- |
| `make env-sync` | Creates `.env` from `.env.example` if missing, or appends any missing keys without overwriting existing values. |

### Environment Setup

| Command | Description |
| --- | --- |
| `make setup` | Full local setup: creates virtual env dependencies, syncs `.env`, initializes ZenML, and installs ZenML integrations. |
| `make .venv` | Installs project dependencies with dev extras using `uv sync --extra dev` (usually invoked by `make setup`). |
| `make zenml-init` | Initializes ZenML in the repo if `.zen` is not present. |
| `make zenml-integrations` | Installs ZenML integrations (`aws`, `s3`, `mlflow`, `evidently`) via uv. |
| `make zenml-connect` | If `ZENML_STORE_API_KEY` is set, uses env-based auth and skips login; otherwise runs `zenml login` against `ZENML_SERVER_URI`. |
| `make zenml-disconnect` | Logs local ZenML client out of the connected ZenML server. |
| `make services-up` | Starts docker-compose services in detached mode. |
| `make services-rebuild` | Rebuilds and starts docker-compose services in detached mode. |
| `make services-down` | Stops and removes docker-compose services. |
| `make services-logs` | Tails docker-compose logs for all services. |
| `make up` | End-to-end local bootstrap: sync env, start services, register local stacks, activate local stack, connect ZenML client. |
| `make rebuild` | Rebuild local services and re-run local stack setup + ZenML connection. |
| `make down` | Stops services and disconnects ZenML client. |

### Code Quality

| Command | Description |
| --- | --- |
| `make lint` | Runs Ruff lint checks and formatting checks. |
| `make fmt` | Auto-fixes lint issues with Ruff and applies formatting. |

### Tests

| Command | Description |
| --- | --- |
| `make test WORKFLOW=<workflow_name>` | Runs unit tests for the selected workflow. |
| `make test-integration WORKFLOW=<workflow_name>` | Runs integration tests for the selected workflow. |
| `make test-all WORKFLOW=<workflow_name>` | Runs all workflow tests with coverage reporting. |

### Workflow Discovery

| Command | Description |
| --- | --- |
| `make list-workflows` | Lists all workflows discoverable by `run.py`. |
| `make list-pipelines WORKFLOW=<workflow_name>` | Lists pipelines available for a workflow. |
| `make validate-workflow-param` | Internal helper target that fails if `WORKFLOW` is not provided. |
| `make validate-pipeline-param` | Internal helper target that fails if `PIPELINE` is not provided. |

### Pipeline Runs (Local)

| Command | Description |
| --- | --- |
| `make run-local-training WORKFLOW=<workflow_name>` | Runs the workflow `training_pipeline` with `configs/local/training_pipeline.yaml` on `local_stack`. |
| `make run-local-serving WORKFLOW=<workflow_name>` | Runs the workflow `serving_pipeline` with `configs/local/serving_pipeline.yaml` on `local_stack`. |
| `make run-local-monitoring WORKFLOW=<workflow_name>` | Runs the workflow `monitoring_pipeline` with `configs/local/monitoring_pipeline.yaml` on `local_stack`. |
| `make run-local-pipeline WORKFLOW=<workflow_name> PIPELINE=<pipeline_name>` | Runs a selected pipeline with `configs/local/<pipeline_name>.yaml` on `local_stack`. |

### Pipeline Runs (AWS)

| Command | Description |
| --- | --- |
| `make run-aws-training WORKFLOW=<workflow_name>` | Runs the workflow `training_pipeline` with `configs/aws/training_pipeline.yaml` on `aws_stack`. |
| `make run-aws-serving WORKFLOW=<workflow_name>` | Runs the workflow `serving_pipeline` with `configs/aws/serving_pipeline.yaml` on `aws_stack`. |
| `make run-aws-monitoring WORKFLOW=<workflow_name>` | Runs the workflow `monitoring_pipeline` with `configs/aws/monitoring_pipeline.yaml` on `aws_stack`. |
| `make run-aws-pipeline WORKFLOW=<workflow_name> PIPELINE=<pipeline_name>` | Runs a selected pipeline with `configs/aws/<pipeline_name>.yaml` on `aws_stack`. |

### Infrastructure

| Command | Description |
| --- | --- |
| `make infra-local` | Registers/configures local ZenML stack components via script. |
| `make infra-aws` | Registers/configures AWS ZenML stack components via script. |

### Stack Selection

| Command | Description |
| --- | --- |
| `make stack-local` | Sets active ZenML stack to `local_stack`. |
| `make stack-aws` | Sets active ZenML stack to `aws_stack`. |

### Cleanup

| Command | Description |
| --- | --- |
| `make clean` | Removes Python cache/build artifacts (`__pycache__`, `.pyc`, `dist`, `build`, `*.egg-info`). |
| `make clean-checkpoints` | Removes local checkpoint directory. |
| `make clean-all` | Runs `clean` and `clean-checkpoints`, then removes `.venv`. |

## Resuming a Failed Training Run

Training checkpoints are saved after every epoch. If the job is interrupted, simply re-run the same command:

```bash
# Re-running after failure automatically resumes from the last completed epoch
make run-local-training WORKFLOW=<workflow_name>
```

Checkpoints are stored in `./checkpoints/<run_id>/` locally or `s3://aips-recs-zenml-checkpoints/<run_id>/` on AWS.

## Key Design Decisions

| Decision                | Choice                                          | Why                                                      |
| ----------------------- | ----------------------------------------------- | -------------------------------------------------------- |
| **Algorithm**           | ALS (not SVD)                                   | Block-parallel, Dask-native, handles implicit feedback   |
| **Checkpointing**       | Epoch-level `.npy` + `.done` marker             | Atomic writes; resume from any epoch failure             |
| **HPO resumability**    | Optuna `load_if_exists=True` + SQLite/PG        | Persists across restarts; no re-running completed trials |
| **Distributed compute** | Dask `LocalCluster` / remote scheduler          | Same code runs locally and on AWS                        |
| **Numba**               | `@njit(parallel=True, nogil=True)` on ALS solve | 5–20× speedup on the per-user least-squares bottleneck   |
