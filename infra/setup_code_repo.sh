#!/usr/bin/env bash
# infra/setup_code_repo.sh
#
# Shared helper to register a GitHub code repository with ZenML.
# Called from infra/local/setup_stacks.sh and infra/aws/setup_stacks.sh
# when GITHUB_TOKEN is set.
#
# Prerequisites:
#   zenml integration install github  (run once; idempotent)
#
# Required environment variables:
#   GITHUB_TOKEN — GitHub Personal Access Token with contents:read on the repo
#
# Optional environment variables:
#   ZENML_CODE_REPO_OWNER  — auto-detected from `git remote` if not set
#   ZENML_CODE_REPO_NAME   — auto-detected from `git remote` if not set
#
# Usage (called by stack setup scripts, not directly):
#   source infra/setup_code_repo.sh

set -euo pipefail

: "${GITHUB_TOKEN:?ERROR: GITHUB_TOKEN is not set}"

# ---------------------------------------------------------------------------
# Auto-detect owner and repo name from the current git remote (origin)
# ---------------------------------------------------------------------------

_detect_git_remote_field() {
  # Extract the remote URL and parse owner/repo from any of:
  #   https://github.com/<owner>/<repo>.git
  #   git@github.com:<owner>/<repo>.git
  local field="$1"  # "owner" or "repo"
  local remote_url
  remote_url="$(git remote get-url origin 2>/dev/null || true)"

  if [[ -z "${remote_url}" ]]; then
    echo ""
    return
  fi

  # Normalise: strip trailing .git, convert SSH to HTTPS-style path
  remote_url="${remote_url%.git}"
  # git@github.com:owner/repo  ->  github.com/owner/repo
  remote_url="$(echo "${remote_url}" | sed 's|git@\([^:]*\):|\1/|')"
  # Now grab the last two path segments: owner/repo
  local owner repo
  owner="$(echo "${remote_url}" | awk -F'/' '{print $(NF-1)}')"
  repo="$(echo "${remote_url}" | awk -F'/' '{print $NF}')"

  if [[ "${field}" == "owner" ]]; then
    echo "${owner}"
  else
    echo "${repo}"
  fi
}

ZENML_CODE_REPO_OWNER="${ZENML_CODE_REPO_OWNER:-$(_detect_git_remote_field owner)}"
ZENML_CODE_REPO_NAME="${ZENML_CODE_REPO_NAME:-$(_detect_git_remote_field repo)}"

if [[ -z "${ZENML_CODE_REPO_OWNER}" || -z "${ZENML_CODE_REPO_NAME}" ]]; then
  echo "  ⚠ Could not auto-detect GitHub owner/repo from git remote."
  echo "    Set ZENML_CODE_REPO_OWNER and ZENML_CODE_REPO_NAME explicitly and re-run."
  exit 1
fi

echo ""
echo "==> Registering GitHub code repository..."
echo "    repo : ${ZENML_CODE_REPO_OWNER}/${ZENML_CODE_REPO_NAME}"

# Ensure the GitHub integration is installed
zenml integration install github --uv -y >/dev/null 2>&1 || true

if zenml code-repository describe "${ZENML_CODE_REPO_NAME}" ``>/dev/null 2>&1; then
  zenml code-repository update "${ZENML_CODE_REPO_NAME}" \
    --owner="${ZENML_CODE_REPO_OWNER}" \
    --repository="${ZENML_CODE_REPO_NAME}" \
    --token="${GITHUB_TOKEN}"
else
  zenml code-repository register "${ZENML_CODE_REPO_NAME}" \
    --type=github \
    --owner="${ZENML_CODE_REPO_OWNER}" \
    --repository="${ZENML_CODE_REPO_NAME}" \
    --token="${GITHUB_TOKEN}"
fi

echo "  ✓ Code repository ready: ${ZENML_CODE_REPO_NAME} (${ZENML_CODE_REPO_OWNER}/${ZENML_CODE_REPO_NAME})"
``