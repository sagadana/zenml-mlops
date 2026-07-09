.PHONY: setup lint test docker-build run-local run-aws clean infra-local infra-aws \
	services-up services-down services-logs up down env-sync zenml-connect

UV := uv
DOCKER_COMPOSE := docker compose

# ── Environment (.env) ────────────────────────────────────────────────────────

# Sync .env from .env.example — adds missing keys, never overwrites existing values
env-sync:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env from .env.example"; \
	else \
		added=0; \
		while IFS= read -r line; do \
			key=$$(echo "$$line" | grep -Eo '^[A-Z_]+' || true); \
			if [ -n "$$key" ] && ! grep -q "^$$key=" .env 2>/dev/null; then \
				echo "$$line" >> .env; \
				added=$$((added+1)); \
			fi; \
		done < .env.example; \
		echo "✓ .env already exists — added $$added missing key(s)"; \
	fi

# Load .env and export all variables (skips blank lines and comments)
-include .env
export

# ── Environment Setup ──────────────────────────────────────────────────────────

setup: .venv env-sync zenml-init zenml-integrations
	@echo "✓ Setup complete. Activate venv: source .venv/bin/activate"

.venv: pyproject.toml
	$(UV) sync --extra dev

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

# Connect local ZenML client to the dockerized ZenML server
zenml-connect:
	$(UV) run zenml login $(ZENML_SERVER_URI) --no-verify-ssl
	@echo "✓ Connected to ZenML server at http://localhost:$(ZENML_SERVER_PORT)"

services-up:
	$(DOCKER_COMPOSE) up -d --build
	@echo " "
	@echo "✓ Local services are up."
	@echo "  --------------------------------------------------- "
	@echo "  ZenML:     http://localhost:$(ZENML_SERVER_PORT)"
	@echo "  MLflow:    http://localhost:$(MLFLOW_TRACKING_PORT)"
	@echo "  Dask UI:   http://localhost:$(DASK_DASHBOARD_PORT)"
	@echo "  --------------------------------------------------- "
	@echo " "

services-down:
	$(DOCKER_COMPOSE) down

services-logs:
	$(DOCKER_COMPOSE) logs -f

up: env-sync services-up infra-local stack-local zenml-connect
	@echo "✓ Local stack configured and connected to ZenML server."

down: services-down

# ── Code Quality ───────────────────────────────────────────────────────────────

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

# ── Tests ──────────────────────────────────────────────────────────────────────

test:
	$(UV) run pytest workflows/$(WORKFLOW)/tests/unit/ -v --tb=short

test-integration:
	$(UV) run pytest workflows/$(WORKFLOW)/tests/integration/ -v --tb=short

test-all:
	$(UV) run pytest workflows/$(WORKFLOW)/tests/ -v --tb=short --cov=workflows --cov-report=term-missing

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
CONFIG_LOCAL := workflows/$(WORKFLOW)/configs/local.yaml
CONFIG_AWS   := workflows/$(WORKFLOW)/configs/aws.yaml

run-local-training: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline training --config $(CONFIG_LOCAL) --stack local_stack

run-local-serving: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline serving --config $(CONFIG_LOCAL) --stack local_stack

run-local-monitoring: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline monitoring --config $(CONFIG_LOCAL) --stack local_stack

run-local-pipeline: validate-workflow-param validate-pipeline-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline $(PIPELINE) --config $(CONFIG_LOCAL) --stack local_stack


# ── Pipeline Runs — AWS ────────────────────────────────────────────────────────

run-aws-training: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline training --config $(CONFIG_AWS) --stack aws_stack

run-aws-serving: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline serving --config $(CONFIG_AWS) --stack aws_stack

run-aws-monitoring: validate-workflow-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline monitoring --config $(CONFIG_AWS) --stack aws_stack

run-aws-pipeline: validate-workflow-param validate-pipeline-param
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline $(PIPELINE) --config $(CONFIG_AWS) --stack aws_stack

# ── Infrastructure ─────────────────────────────────────────────────────────

infra-local:
	bash infra/local/setup_stacks.sh

infra-aws:
	bash infra/aws/setup_stacks.sh

# ── Stack ────────────────────────────────────────────────────────────────────

stack-local:
	$(UV) run zenml stack set local_stack

stack-aws:
	$(UV) run zenml stack set aws_stack

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/

clean-checkpoints:
	rm -rf checkpoints/

clean-all: clean clean-checkpoints
	rm -rf .venv
