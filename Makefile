.PHONY: init

UV := uv
DOCKER_COMPOSE := docker compose

# ── Environment (.env) ────────────────────────────────────────────────────────

# Ensure .env exists by copying from .env.example if missing
.env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env from .env.example"; \
	else \
		echo "✓ .env already exists — keeping existing file unchanged"; \
	fi

# Load .env and export all variables (skips blank lines and comments)
-include .env
export

# ── Environment Setup ──────────────────────────────────────────────────────────

.venv: pyproject.toml
	$(UV) sync --extra dev
	$(UV) run python -m ensurepip --upgrade
	@echo "✓ Virtual environment created and dependencies installed"

zenml-init:
	@if [ ! -d ".zen" ]; then \
		$(UV) run zenml init; \
		echo "✓ ZenML initialized"; \
	else \
		echo "✓ ZenML already initialized"; \
	fi
	
zenml-integrations:
	$(UV) run zenml integration install aws s3 mlflow evidently --uv -y
	@echo "✓ ZenML integrations installed"

zenml-service-account:
	$(UV) run bash infra/setup_service_account.sh

zenml-default-project:
	$(UV) run zenml project set default; 

# Connect local ZenML client to the dockerized ZenML server
zenml-connect:
	@if [ -z $$ZENML_STORE_API_KEY ]; then \
		echo "✓ ZENML_STORE_API_KEY exists in environment; skipping zenml login"; \
	else \
		$(UV) run zenml login $(ZENML_SERVER_URI) --no-verify-ssl; \
	fi
	@echo "✓ Connected to ZenML server at http://localhost:$(ZENML_SERVER_PORT)"

# Reconnect local ZenML client to the dockerized ZenML server (useful if facing authentication issues)
zenml-reconnect:
	$(UV) run zenml logout
	$(UV) run zenml login $(ZENML_SERVER_URI) --refresh --no-verify-ssl
	@echo "✓ Reconnected to ZenML server at http://localhost:$(ZENML_SERVER_PORT)"

# Disconnect local ZenML client from the dockerized ZenML server
zenml-disconnect:
	@if [ -z $$ZENML_STORE_API_KEY ]; then \
		echo "✓ ZENML_STORE_API_KEY exists in environment; skipping zenml logout"; \
	else \
		$(UV) run zenml logout; \
	fi
	@echo "✓ Disconnected from ZenML server"

services-up:
	$(DOCKER_COMPOSE) up -d
	@echo " "
	@echo "✓ Local services are up."
	@echo "  ------------------------------------------------------------------ "
	@echo "  ZenML:     			http://localhost:$(ZENML_SERVER_PORT)"
	@echo "  MLflow:    			http://localhost:$(MLFLOW_TRACKING_PORT)"
	@echo "  SeaweedFS: 			http://localhost:$(SEAWEEDFS_S3_PORT)"
	@echo "  SeaweedFS Admin UI:	http://localhost:$(SEAWEEDFS_ADMIN_PORT)"
	@echo "  ------------------------------------------------------------------ "
	@echo " "

	# Wait for services to be fully up and running
	@sleep 6 

services-rebuild:
	$(DOCKER_COMPOSE) up -d --build
	@echo " "
	@echo "✓ Local services are up."
	@echo "  ------------------------------------------------------------------ "
	@echo "  ZenML:     			http://localhost:$(ZENML_SERVER_PORT)"
	@echo "  MLflow:    			http://localhost:$(MLFLOW_TRACKING_PORT)"
	@echo "  SeaweedFS: 			http://localhost:$(SEAWEEDFS_S3_PORT)"
	@echo "  SeaweedFS Admin UI:	http://localhost:$(SEAWEEDFS_ADMIN_PORT)"
	@echo "  ------------------------------------------------------------------ "
	@echo " "

	# Wait for services to be fully up and running
	@sleep 10

services-down:
	$(DOCKER_COMPOSE) down

services-logs:
	$(DOCKER_COMPOSE) logs -f

init: .env services-rebuild zenml-reconnect zenml-init zenml-integrations zenml-default-project infra-local stack-local
	@echo "✓ Local stack initialized and connected to ZenML server."

up: services-up zenml-connect zenml-init zenml-integrations zenml-default-project infra-local stack-local
	@echo "✓ Local stack configured and connected to ZenML server."

rebuild: clean .env services-rebuild zenml-reconnect zenml-init zenml-integrations zenml-default-project infra-local stack-local
	@echo "✓ Local stack rebuilt and connected to ZenML server."

down: services-down zenml-disconnect
	@echo "✓ Local services stopped and disconnected from ZenML server."


# ── Code Quality ───────────────────────────────────────────────────────────────

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

# -- Available Workflows & Pipelines --------------------------------------------------------

list-workflows:
	$(UV) run python run.py list-workflows

list-pipelines:
	$(UV) run python run.py list-pipelines --workflow $(WORKFLOW)

validate-workflow-param:
	@if [ -z "$(WORKFLOW)" ]; then \
		echo "Error: WORKFLOW is not set. Please specify a workflow name, e.g., WORKFLOW=my_workflow"; \
		exit 1; \
	fi

validate-pipeline-param:
	@if [ -z "$(PIPELINE)" ]; then \
		echo "Error: PIPELINE is not set. Please specify a pipeline name, e.g., PIPELINE=training"; \
		exit 1; \
	fi

# ── Pipeline Runs ──────────────────────────────────────────────────────────────
CONFIG_LOCAL_DIR := workflows/$(WORKFLOW)/configs/local
CONFIG_AWS_DIR   := workflows/$(WORKFLOW)/configs/aws

CONFIG_LOCAL_DATA   := $(CONFIG_LOCAL_DIR)/data_pipeline.yaml
CONFIG_LOCAL_TRAINING   := $(CONFIG_LOCAL_DIR)/training_pipeline.yaml
CONFIG_LOCAL_BATCH_INFERENCE := $(CONFIG_LOCAL_DIR)/batch_inference_pipeline.yaml
CONFIG_LOCAL_DEPLOYMENT := $(CONFIG_LOCAL_DIR)/deployment_pipeline.yaml
CONFIG_LOCAL_MONITORING := $(CONFIG_LOCAL_DIR)/monitoring_pipeline.yaml
CONFIG_LOCAL_PIPELINE   := $(CONFIG_LOCAL_DIR)/$(PIPELINE).yaml

CONFIG_AWS_TRAINING   := $(CONFIG_AWS_DIR)/training_pipeline.yaml
CONFIG_AWS_BATCH_INFERENCE := $(CONFIG_AWS_DIR)/batch_inference_pipeline.yaml
CONFIG_AWS_DEPLOYMENT := $(CONFIG_AWS_DIR)/deployment_pipeline.yaml
CONFIG_AWS_MONITORING := $(CONFIG_AWS_DIR)/monitoring_pipeline.yaml
CONFIG_AWS_PIPELINE   := $(CONFIG_AWS_DIR)/$(PIPELINE).yaml

run-local-data: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline data_pipeline --config $(CONFIG_LOCAL_DATA) --stack $(ZENML_LOCAL_STACK_NAME)

run-local-training: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline training_pipeline --config $(CONFIG_LOCAL_TRAINING) --stack $(ZENML_LOCAL_STACK_NAME)

run-local-serving: run-local-deployment

run-local-batch-inference: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline batch_inference_pipeline --config $(CONFIG_LOCAL_BATCH_INFERENCE) --stack $(ZENML_LOCAL_STACK_NAME)

run-local-deployment: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline deployment_pipeline --config $(CONFIG_LOCAL_DEPLOYMENT) --stack $(ZENML_LOCAL_STACK_NAME)

run-local-monitoring: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline monitoring_pipeline --config $(CONFIG_LOCAL_MONITORING) --stack $(ZENML_LOCAL_STACK_NAME)

run-local-pipeline: validate-workflow-param validate-pipeline-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline $(PIPELINE) --config $(CONFIG_LOCAL_PIPELINE) --stack $(ZENML_LOCAL_STACK_NAME)


# ── Pipeline Runs — AWS ────────────────────────────────────────────────────────

run-aws-training: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline training_pipeline --config $(CONFIG_AWS_TRAINING) --stack aws_stack

run-aws-serving: run-aws-deployment

run-aws-batch-inference: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline batch_inference_pipeline --config $(CONFIG_AWS_BATCH_INFERENCE) --stack aws_stack

run-aws-deployment: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline deployment_pipeline --config $(CONFIG_AWS_DEPLOYMENT) --stack aws_stack

run-aws-monitoring: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline monitoring_pipeline --config $(CONFIG_AWS_MONITORING) --stack aws_stack

run-aws-pipeline: validate-workflow-param validate-pipeline-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline $(PIPELINE) --config $(CONFIG_AWS_PIPELINE) --stack aws_stack

# ── Infrastructure ─────────────────────────────────────────────────────────

infra-local:
	$(UV) run bash infra/local/setup_stacks.sh

infra-aws:
	$(UV) run bash infra/aws/setup_stacks.sh

# ── Stack ────────────────────────────────────────────────────────────────────

stack-local:
	$(UV) run zenml stack set $(ZENML_LOCAL_STACK_NAME)

stack-aws:
	$(UV) run zenml stack set aws_stack

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	@echo "Cleaning up build artifacts, caches, and checkpoints..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf dist/ build/ *.egg-info/ .zen/ checkpoints/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .cache/

clean-all: clean clean-checkpoints
	@echo "Cleaning up virtual environment..."
	@rm -rf .venv
