.PHONY: setup lint test docker-build run-local run-aws clean infra-local infra-aws \
	services-up services-down services-logs zenml-up zenml-down

UV := uv
DOCKER_COMPOSE := docker compose
MLFLOW_TRACKING_URI ?= http://localhost:5000

# ── Environment Setup ──────────────────────────────────────────────────────────

setup: .venv zenml-init zenml-integrations
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
	$(UV) run zenml integration install aws s3 mlflow --uv -y
	@echo "✓ ZenML integrations installed"

services-up:
	$(DOCKER_COMPOSE) up -d --build
	@echo "✓ Local services are up."
	@echo "  ZenML:     http://localhost:8237"
	@echo "  MLflow:    http://localhost:5000"
	@echo "  Evidently: http://localhost:8000"
	@echo "  Dask UI:   http://localhost:8787"

services-down:
	$(DOCKER_COMPOSE) down

services-logs:
	$(DOCKER_COMPOSE) logs -f

zenml-up: services-up infra-local stack-local
	@echo "✓ Local stack configured against compose services."

zenml-down: services-down

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

# ── Pipeline Runs ──────────────────────────────────────────────────────────────
# Override on the command line: make run-local-training WORKFLOW=my_workflow
WORKFLOW     := matrix_factorization
PIPELINE     := training
CONFIG_LOCAL := workflows/$(WORKFLOW)/configs/local.yaml
CONFIG_AWS   := workflows/$(WORKFLOW)/configs/aws.yaml

run-local-training:
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline training --config $(CONFIG_LOCAL) --stack local_stack

run-local-serving:
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline serving --config $(CONFIG_LOCAL) --stack local_stack

run-local-monitoring:
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline monitoring --config $(CONFIG_LOCAL) --stack local_stack

run-local-pipeline:
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline $(PIPELINE) --config $(CONFIG_LOCAL) --stack local_stack


# ── Pipeline Runs — AWS ────────────────────────────────────────────────────────

run-aws-training:
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline training --config $(CONFIG_AWS) --stack aws_stack

run-aws-serving:
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline serving --config $(CONFIG_AWS) --stack aws_stack

run-aws-monitoring:
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline monitoring --config $(CONFIG_AWS) --stack aws_stack

run-aws-pipeline:
	$(UV) run python run.py run --workflow $(WORKFLOW) --pipeline $(PIPELINE) --config $(CONFIG_AWS) --stack aws_stack

# ── Docker ─────────────────────────────────────────────────────────────────────

docker-build:
	docker build -t aips-zenml-$(WORKFLOW):latest -f docker/pipeline/Dockerfile .

docker-build-serving:
	docker build -t aips-zenml-$(WORKFLOW)-serving:latest -f docker/serving/Dockerfile --build-arg WORKFLOW=$(WORKFLOW) .

# ── Infrastructure ─────────────────────────────────────────────────────────

infra-local:
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) bash infra/local/setup_stacks.sh

infra-aws:
	bash infra/aws/setup_stacks.sh

# ── Stack ────────────────────────────────────────────────────────────────────

stack-local:
	$(UV) run zenml stack set local_stack

stack-aws:
	$(UV) run zenml stack set aws_stack

# ── Serving ────────────────────────────────────────────────────────────────────

serve-local:
	$(UV) run uvicorn workflows.$(WORKFLOW).serving.app:app --host 0.0.0.0 --port 8080 --reload

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/

clean-checkpoints:
	rm -rf checkpoints/

clean-all: clean clean-checkpoints
	rm -rf .venv
