# AIPS Recommendations — ZenML MLOps Platform

End-to-end MLOps platform built on ZenML. Runs locally or on AWS with a single config switch.

## Architecture

```mermaid
graph LR
    D[Dataset] --> DP[data_pipeline]
    DP --> HP[hpo_pipeline]
    DP --> TP[training_pipeline]
    HP --> TP
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
# Equivalent to: uv sync --extra dev

# 3. List available workflows and pipelines
make list-workflows
make list-pipelines WORKFLOW=<workflow_name>

# 4. Run pipeline
make run-local-pipeline WORKFLOW=<workflow_name> PIPELINE=<pipeline_name>

# 4. Start serving API
make serve-local WORKFLOW=<workflow_name>
# Test: curl -X POST http://localhost:8080/recommend \
#        -H "Content-Type: application/json" \
#        -d '{"user_id": 1, "top_k": 10}'
```

## AWS Deployment

```bash
# 1. Set environment variables
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=us-east-1
export ZENML_EXECUTION_ROLE_ARN=arn:aws:iam::123456789012:role/zenml-execution-role
export MLFLOW_TRACKING_URI=http://<your-ec2-ip>:5000

# 2. Deploy ZenML server to AWS (EC2 + RDS PostgreSQL)
zenml deploy --provider aws --region us-east-1

# 3. Provision S3, ECR, IAM + register ZenML stacks
bash infra/aws/setup_stacks.sh

# 4. Switch to AWS stack
zenml stack set aws_stack

# 5. Run full pipeline
make run-aws-data WORKFLOW=<workflow_name>
make run-aws-hpo WORKFLOW=<workflow_name>       # optional
make run-aws-training WORKFLOW=<workflow_name>
make run-aws-serving WORKFLOW=<workflow_name>
make run-aws-monitoring WORKFLOW=<workflow_name>
```

## Pipeline Reference

| Pipeline                     | Command                                              | Description                          |
| ---------------------------- | ---------------------------------------------------- | ------------------------------------ |
| `<workflow_name>-data`       | `make run-local-data WORKFLOW=<workflow_name>`       | Download → validate → encode → split |
| `<workflow_name>-hpo`        | `make run-local-hpo WORKFLOW=<workflow_name>`        | Optuna HPO (optional, resumable)     |
| `<workflow_name>-training`   | `make run-local-training WORKFLOW=<workflow_name>`   | Train → evaluate → register          |
| `<workflow_name>-serving`    | `make run-local-serving WORKFLOW=<workflow_name>`    | Batch recs + real-time endpoint      |
| `<workflow_name>-monitoring` | `make run-local-monitoring WORKFLOW=<workflow_name>` | Drift detection + retrain trigger    |

## Configuration

All environment differences are controlled by config files — no code changes needed:

| Config                                         | Dask Partitions | HPO Storage | Checkpoint Path               |
| ---------------------------------------------- | --------------- | ----------- | ----------------------------- |
| `workflows/<workflow_name>/configs/local.yaml` | 4 (default)     | SQLite      | `./checkpoints`               |
| `workflows/<workflow_name>/configs/aws.yaml`   | 64 (default)    | PostgreSQL  | `s3://aips-zenml-checkpoints` |

## Adding a New Pipeline

See [AGENTS.md](AGENTS.md) and [.agents/skills/create-e2e-ml-workflow/SKILL.md](.agents/skills/create-e2e-ml-workflow/SKILL.md) for the full guide.

## Running Tests

```bash
make test              # unit tests only
make test-integration  # full pipeline integration tests
make test-all          # everything with coverage
```

## Resuming a Failed Training Run

Training checkpoints are saved after every epoch. If the job is interrupted, simply re-run the same command:

```bash
# Re-running after failure automatically resumes from the last completed epoch
make run-local-training WORKFLOW=<workflow_name>
```

Checkpoints are stored in `./checkpoints/<run_id>/` locally or `s3://aips-zenml-checkpoints/<run_id>/` on AWS.

## Key Design Decisions

| Decision                | Choice                                          | Why                                                      |
| ----------------------- | ----------------------------------------------- | -------------------------------------------------------- |
| **Algorithm**           | ALS (not SVD)                                   | Block-parallel, Dask-native, handles implicit feedback   |
| **Checkpointing**       | Epoch-level `.npy` + `.done` marker             | Atomic writes; resume from any epoch failure             |
| **HPO resumability**    | Optuna `load_if_exists=True` + SQLite/PG        | Persists across restarts; no re-running completed trials |
| **Distributed compute** | Dask `LocalCluster` / remote scheduler          | Same code runs locally and on AWS                        |
| **Numba**               | `@njit(parallel=True, nogil=True)` on ALS solve | 5–20× speedup on the per-user least-squares bottleneck   |
