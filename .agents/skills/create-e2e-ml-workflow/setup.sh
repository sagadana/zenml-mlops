#!/usr/bin/env bash
# Usage: WF=workflows/<workflow_name> bash .agents/skills/create-e2e-ml-workflow/setup.sh
set -euo pipefail

: "${WF:?Set WF=workflows/<workflow_name> before running this script}"

mkdir -p "$WF/configs/local"
mkdir -p "$WF/configs/aws"
mkdir -p "$WF/materializers"
mkdir -p "$WF/models"
mkdir -p "$WF/pipelines"
mkdir -p "$WF/serving"
mkdir -p "$WF/steps/data_ingestion"
mkdir -p "$WF/steps/data_validation"
mkdir -p "$WF/steps/feature_engineering"
mkdir -p "$WF/steps/hpo"
mkdir -p "$WF/steps/training"
mkdir -p "$WF/steps/model_evaluation"
mkdir -p "$WF/steps/serving"
mkdir -p "$WF/utils"

touch workflows/__init__.py
touch "$WF/__init__.py"
touch "$WF/configs/__init__.py"
touch "$WF/materializers/__init__.py"
touch "$WF/models/__init__.py"
touch "$WF/pipelines/__init__.py"
touch "$WF/serving/__init__.py"
touch "$WF/steps/__init__.py"
touch "$WF/utils/__init__.py"

echo "Directory structure created under $WF"
