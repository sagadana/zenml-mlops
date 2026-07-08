#!/usr/bin/env bash
# infra/local/setup_stacks.sh
#
# Idempotent ZenML stack setup script for local development.
#
# Usage:
#   bash infra/local/setup_stacks.sh

set -euo pipefail

echo "==> Registering local ZenML stack..."

zenml stack describe local_stack 2>/dev/null || \
  zenml stack register local_stack \
    --orchestrator=default \
    --artifact-store=default
echo "  ✓ Stack: local_stack"

echo ""
echo "=== Local setup complete ==="
echo "Available stacks:"
zenml stack list
echo ""
echo "To switch stack:"
echo "  zenml stack set local_stack"
