---
name: sync-agent-docs
description: Keeps AGENTS.md, README command references, skill definitions (SKILL.md, stubs, setup.sh), workflow specs (.agents/specs/**), and repo structure documentation in sync with actual workflow implementations and Makefile targets. Use when a reference workflow changes, when stubs drift from the implementation, when Makefile commands change, when a new create-e2e-* skill is added, or as a periodic hygiene pass.
---

# Sync Agent Docs and Skills

## Overview

Workflow implementations evolve, but the documentation and stubs that agents rely on don't update automatically. This skill performs a structured audit-and-update pass over five layers:

1. **`AGENTS.md`** — repo structure block, persona `Files to know` lists, key conventions
2. **`README.md`** — Make command lists and command examples that must match the Makefile
3. **`create-e2e-*` skills** — `SKILL.md` step paths/conventions, all stubs, `setup.sh`
4. **`.agents/specs/**`** — per-workflow design documents (`wf_<workflow_name>.md`)
5. **Other skills** that contain file path references to workflow files

All layers are derived from the same **reference workflows** (the concrete production implementations) plus the current `Makefile` command surface. This skill keeps them in sync.

---

## Reference Workflow Map

Each `create-e2e-*` skill is templated from one concrete reference workflow. Update this table whenever a new skill or workflow is added.

| Skill directory | Reference workflow |
|---|---|
| `create-e2e-ml-workflow` | `workflows/matrix_factorization` |

> **Adding a new `create-e2e-*` skill?** See [Adding a New Skill](#adding-a-new-create-e2e-skill) at the bottom of this file before running the sync.

---

## When to Use

- A reference workflow implementation changed (new step, renamed parameter, new import, new file)
- A stub was found to produce incorrect code when used by an agent
- A new `create-e2e-*` skill was created and needs to be validated against its reference
- The repo structure changed (directory added/removed, Dockerfile moved, helpers extracted)
- A workflow spec (`.agents/specs/wf_*.md`) is out of date with the implementation
- Makefile targets were added/renamed/removed and command lists in docs may be stale
- Periodic hygiene (run after any major feature ship to catch drift)

---

## Step 0: Discover Scope

### 0a. Find all create-e2e-* skills

```bash
ls .agents/skills/ | grep '^create-e2e-'
```

Cross-check each against the [Reference Workflow Map](#reference-workflow-map). Any skill not in the map needs a row added before proceeding — read its `SKILL.md` preamble to infer which workflow it templates from.

### 0b. Find all reference workflows

```bash
find workflows -maxdepth 1 -mindepth 1 -type d | sort
```

Each directory with a `__init__.py` and a `pipelines/` subdirectory is a candidate reference workflow.

### 0c. Clarify scope with the user (or infer from recent changes)

Ask:
- Is this a **full sweep** (all skills + AGENTS.md)?
- Or a **targeted sync** for one skill/workflow?

For targeted syncs, jump directly to the relevant step.

---

## Step 1: Audit AGENTS.md

### 1a. Repository Structure Block

Read the fenced code block under `## Repository Structure` in `AGENTS.md`.

Compare against the actual tree:

```bash
# Top-level directories
ls -1

# Workflow internals
find workflows -maxdepth 2 -type d | sort

# Docker assets
find docker -maxdepth 2 -type f | sort

# Shared helpers
ls helpers/
```

Update the structure block if:
- A top-level directory was added or removed (e.g. `helpers/`, `infra/`, `tests/`)
- A workflow's internal layout changed (subdirectory added/removed)
- A per-workflow Dockerfile was removed in favour of a shared one under `docker/`
- The `docker/` structure changed

**Do not** make the structure block exhaustive — it is a representative sample. Depth ≤ 3 levels. Omit `__pycache__`, `.venv`, `.git`.

### 1b. Persona `Files to know` sections

For each persona block (DataEngineer, MLEngineer, MLOpsEngineer, ServingEngineer):

```bash
# Verify each listed path still exists
rg -o 'workflows/[^\s`]+\.py' AGENTS.md | sort -u | xargs -I{} sh -c '[ -f "{}" ] && echo "OK: {}" || echo "MISSING: {}"'
rg -o 'helpers/[^\s`]+\.py' AGENTS.md | sort -u | xargs -I{} sh -c '[ -f "{}" ] && echo "OK: {}" || echo "MISSING: {}"'
```

Update entries for:
- Files that have been moved or renamed → update path
- Files that have been deleted → remove the entry
- New critical files added to a reference workflow (e.g. new utility, new shared step) → add an entry

### 1c. Key Conventions

Re-read the Key Conventions numbered list. Verify each convention is still accurate by spot-checking the reference workflows:

```bash
# Convention: Never use 'pipeline' as variable name — spot check
grep -rn '\bpipeline\s*=' workflows/ --include="*.py" | grep -v "def \|@\|#"

# Convention: Annotated outputs — spot check
grep -rn "def.*->.*tuple\|def.*->.*Annotated" workflows/ --include="*.py" | head -5

# Convention: Absolute imports from repo root
grep -rn "^from \." workflows/ --include="*.py" | head -5
```

Add new conventions if the reference workflow has established a new project-wide pattern.
Remove or revise conventions that are no longer accurate.

### 1d. Audit command examples in AGENTS.md

Verify that command examples in AGENTS.md still map to valid Make targets:

```bash
# List declared Make targets
awk -F':' '/^[a-zA-Z0-9_.-]+:/ {print $1}' Makefile | sort -u

# Extract make commands from AGENTS.md for quick manual verification
rg "make " AGENTS.md
```

Update AGENTS.md examples when:
- A referenced Make target was renamed/removed
- Required parameters changed (`WORKFLOW`, `PIPELINE`, etc.)
- New canonical command names should replace old examples

---

## Step 1.5: Audit README Command Lists

### 1.5a. Verify README command sections match Makefile

When README contains a command catalog (e.g., "Make Commands Reference"), ensure:
- Every documented target exists in `Makefile`
- Every user-facing target in `Makefile` is documented
- Targets are grouped by the same functional sections used in `Makefile`
- Descriptions remain short, accurate, and parameterized with placeholders

Useful checks:

```bash
# Canonical list of Make targets
awk -F':' '/^[a-zA-Z0-9_.-]+:/ {print $1}' Makefile | sort -u

# All make commands currently documented in README
rg "make [a-zA-Z0-9_.-]+" README.md
```

### 1.5b. Keep command examples coherent across docs

Cross-check command usage in:
- `README.md`
- `AGENTS.md`
- Relevant skill docs under `.agents/skills/**/SKILL.md`

If one document updates command names/parameters, propagate to the others in the same sync pass.

---

## Step 2: Audit Each `create-e2e-*` Skill

Repeat this section for every skill found in Step 0a.

### 2a. Map stub files to reference files

The canonical mapping is:

| Stub path (relative to skill root) | Reference path (relative to repo root) |
|---|---|
| `stubs/configs/local.yaml.stub` | `workflows/<ref>/configs/local.yaml` |
| `stubs/configs/aws.yaml.stub` | `workflows/<ref>/configs/aws.yaml` |
| `stubs/materializers/model_materializer.py.stub` | `workflows/<ref>/materializers/<algo>_materializer.py` |
| `stubs/materializers/dask_dataframe_materializer.py.stub` | `workflows/<ref>/materializers/dask_dataframe_materializer.py` |
| `stubs/materializers/__init__.py.stub` | `workflows/<ref>/materializers/__init__.py` |
| `stubs/models/workflow_model.py.stub` | `workflows/<ref>/models/<algo>.py` |
| `stubs/models/__init__.py.stub` | `workflows/<ref>/models/__init__.py` |
| `stubs/steps/data_ingestion/ingest.py.stub` | `workflows/<ref>/steps/data_ingestion/ingest.py` |
| `stubs/steps/data_validation/validate.py.stub` | `workflows/<ref>/steps/data_validation/validate.py` |
| `stubs/steps/feature_engineering/encoders.py.stub` | `workflows/<ref>/steps/feature_engineering/encoders.py` |
| `stubs/steps/feature_engineering/split.py.stub` | `workflows/<ref>/steps/feature_engineering/split.py` |
| `stubs/steps/hpo/run_hpo.py.stub` | `workflows/<ref>/steps/hpo/run_hpo.py` |
| `stubs/steps/training/train.py.stub` | `workflows/<ref>/steps/training/train.py` |
| `stubs/steps/model_evaluation/evaluate.py.stub` | `workflows/<ref>/steps/model_evaluation/evaluate.py` |
| `stubs/steps/model_evaluation/register.py.stub` | `workflows/<ref>/steps/model_evaluation/register.py` |
| `stubs/steps/serving/batch_predict.py.stub` | `workflows/<ref>/steps/serving/batch_predict.py` |
| `stubs/steps/serving/build_image.py.stub` | `workflows/<ref>/steps/serving/build_image.py` |
| `stubs/steps/serving/deploy.py.stub` | `workflows/<ref>/steps/serving/deploy.py` |
| `stubs/pipelines/training_pipeline.py.stub` | `workflows/<ref>/pipelines/training_pipeline.py` |
| `stubs/pipelines/serving_pipeline.py.stub` | `workflows/<ref>/pipelines/serving_pipeline.py` |
| `stubs/pipelines/monitoring_pipeline.py.stub` | `workflows/<ref>/pipelines/monitoring_pipeline.py` |
| `stubs/pipelines/__init__.py.stub` | `workflows/<ref>/pipelines/__init__.py` |
| `stubs/serving/app.py.stub` | `workflows/<ref>/serving/app.py` |
| `stubs/serving/__init__.py.stub` | `workflows/<ref>/serving/__init__.py` |
| `stubs/tests/unit/test_workflow_model.py.stub` | `workflows/<ref>/tests/unit/test_<algo>.py` |

### 2b. Diff each stub against its reference

For each stub/reference pair, check for **structural divergence** — changes that apply to all workflows, not just the reference's specific algorithm.

Update the stub when the reference shows:

| Signal | What to update in stub |
|---|---|
| New import added (e.g. `psutil`, `boto3`) | Add the import; replace algorithm-specific usage with a placeholder comment |
| New function parameter | Add parameter to stub function signature with placeholder default |
| New `@step` decorator option (e.g. new `output_materializers` key) | Mirror in stub |
| Config YAML gains a new top-level key or step entry | Add to YAML stub with placeholder value |
| `to_parquet` / serialization call gains new argument | Mirror the argument |
| Pipeline step gains new output wiring or step | Mirror in pipeline stub |
| Serving app `/health` response gains new fields | Add fields to stub's response model |
| Batch step switches from per-user to batched iteration | Update stub iteration pattern |
| New pipeline added to reference workflow | Add corresponding stub |
| Model class gains new property or method | Add stub method with `raise NotImplementedError(...)` |

**Never do** when updating stubs:
- Replace `<workflow_name>`, `<ModelClassName>`, `<model_name>`, `<WorkflowName>` with reference-specific values
- Copy algorithm-specific logic (ALS solver calls, dataset-specific parsers, MovieLens URLs, etc.)
- Replace `raise ValueError("Customize ...")` stubs in implementation bodies with real logic
- Add reference-workflow-specific hyperparameter ranges to the HPO stub

### 2c. Update SKILL.md step content

Read the skill's `SKILL.md` and verify:

- **File paths in step headers** — every `### \`workflows/<workflow_name>/steps/...\`` path must match the reference layout
- **Stub references** — every `> **Stub:** [stubs/...]` link must point to a file that actually exists
- **Critical Conventions list** — add new conventions discovered from the reference; remove stale ones
- **Step descriptions** — if a step was renamed or a new step was added to the workflow, update the corresponding SKILL.md section

### 2d. Update setup.sh

```bash
# Reference directory tree
find workflows/<ref_workflow> -type d | sort

# Current setup.sh dirs
grep "mkdir -p" .agents/skills/<skill>/setup.sh
```

Add missing `mkdir -p` lines. Add missing `touch` lines for `__init__.py` files. Remove lines for directories that no longer exist in the reference.

### 2e. Verify all stub references resolve

```bash
rg -o 'stubs/[^\)]+\.stub' .agents/skills/<skill>/SKILL.md | while read f; do
  [ -f ".agents/skills/<skill>/$f" ] && echo "OK: $f" || echo "MISSING: $f"
done
```

Fix any broken paths found.

---

## Step 3: Audit `.agents/specs/**`

Each workflow has a companion spec file at `.agents/specs/wf_<workflow_name>.md` that documents confirmed design decisions, architecture diagrams, and step-level implementation details.

### 3a. Discover spec files

```bash
ls .agents/specs/
```

Every workflow directory under `workflows/` should have a corresponding spec file. If a workflow exists without a spec, flag it — a new spec may need to be written (out of scope for this skill; raise with the user).

### 3b. Check spec accuracy against the implementation

For each spec file, verify the following sections against the actual workflow code:

| Spec section | What to check |
|---|---|
| **Confirmed Decisions table** | Algorithm, serving mode, dataset names, monitoring tool — still accurate? |
| **Architecture diagram** (`mermaid` block) | Pipeline step names match `@pipeline` and `@step` definitions in `workflows/<wf>/pipelines/` and `workflows/<wf>/steps/` |
| **Step-level descriptions** | Step names, input/output types, config parameter names — still match the implementation |
| **Config parameter names** | YAML keys mentioned in the spec match the actual `configs/local.yaml` and `configs/aws.yaml` |
| **AWS component names** | Stack component names match `infra/aws/setup_stacks.sh` |
| **File/path references** | Every file path mentioned in the spec still exists |

```bash
# Verify step names in the spec match pipeline definitions
rg '^def\s+\w+\(' workflows/<workflow_name>/steps workflows/<workflow_name>/pipelines -g '*.py'

# Verify config keys mentioned in spec exist in configs
rg -o '`[a-z_]+`' .agents/specs/wf_<workflow_name>.md | sort -u
```

### 3c. Update the spec

Update the spec when:
- A step was renamed → update all references in the architecture diagram and step descriptions
- A step was added or removed → update the `mermaid` diagram and add/remove the corresponding section
- A config parameter was renamed or added → update the relevant table or code block
- A confirmed decision changed (e.g., serving mode switched) → update the Confirmed Decisions table and rationale
- An AWS component name changed → update the stack components table

**Do not** rewrite the rationale column of the Confirmed Decisions table unless the decision itself changed — rationale is intentional prose, not derived from code.

---

## Step 4: Audit Other Skills That Reference Workflow Files

```bash
# Find any skill SKILL.md that contains a workflow or helper file path
grep -rl "workflows/\|helpers/" .agents/skills/*/SKILL.md | grep -v create-e2e-
```

For each file found:
- Check if the referenced paths still exist
- Update broken paths

---

## Step 5: Commit

```bash
git add AGENTS.md \
        .agents/skills/sync-agent-docs/SKILL.md \
        .agents/skills/*/SKILL.md \
        .agents/skills/*/stubs/ \
        .agents/skills/*/setup.sh \
        .agents/specs/
git status  # review before committing

git commit -m "docs: sync agent docs and skill stubs with current implementations

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## What NOT to Change

| Item | Reason |
|---|---|
| Algorithm-specific implementation bodies inside stubs | Stubs are templates, not reference copies |
| Stub placeholder variables (`<workflow_name>` etc.) | Intentional — replaced at workflow creation time |
| AGENTS.md persona command examples (`make run-local-training`) | Only change if the `make` target itself changed |
| README Make command catalog groupings | Do not reorganize stylistically unless Makefile section structure changed |
| Skill `description:` frontmatter of other skills | Only change if the skill's purpose fundamentally changed |
| The Reference Workflow Map in this file | Only change when adding/removing skills — do not rename existing mappings |
| Rationale column in spec Confirmed Decisions tables | Only change if the underlying decision changed, not just because wording could be improved |

---

## Adding a New `create-e2e-*` Skill

When a brand-new workflow type is added to the repo and needs its own creation skill:

1. **Create the skill directory**:
   ```bash
   mkdir -p .agents/skills/create-e2e-<type>-workflow/stubs
   ```

2. **Write `SKILL.md`** — copy `create-e2e-ml-workflow/SKILL.md` as a starting template; replace all `matrix_factorization`/`ALS`/`MovieLens` references with the new workflow's specifics.

3. **Write `setup.sh`** — copy from `create-e2e-ml-workflow/setup.sh`; add/remove directory entries to match the new workflow's layout.

4. **Populate stubs** — copy each stub from `create-e2e-ml-workflow/stubs/`; replace algorithm-specific bodies with `raise ValueError("Customize ...")` placeholders while keeping the structural scaffold.

5. **Add a row to the [Reference Workflow Map](#reference-workflow-map)** in this file:
   ```markdown
   | `create-e2e-<type>-workflow` | `workflows/<ref_workflow_dir>` |
   ```

6. **Run this skill** (Step 2 only, scoped to the new skill) to validate all stubs match the reference from day one.

7. **Update `AGENTS.md`** — add the new skill to the "Creating a New Pipeline" section if its workflow type is substantively different from existing ones.
