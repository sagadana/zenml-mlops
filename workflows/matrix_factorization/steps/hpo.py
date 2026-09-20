"""
steps/hpo/run_hpo.py

ZenML steps: run_hpo_trial, collect_best_hpo_params

Hyperparameter optimization for the ALS model using Optuna and ZenML artifacts.

An in-memory Optuna study samples all trial configurations. ZenML maps one
training step over those configurations and passes their result artifacts to a
fan-in step that selects the best hyperparameters. No external Optuna storage
is required.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

import optuna
import pandas as pd
from pydantic import BaseModel
from zenml import log_metadata, step
from zenml.client import Client

from workflows.matrix_factorization.models.base_recommender import (
    BaseRecommender,
    EpochState,
    Hyperparameters,
    load_recommender_class,
)

logger = logging.getLogger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker

optuna.logging.set_verbosity(logger.level)

type HPOMetric = Literal["loss", "precision", "recall", "ndcg"]

HPO_SPACES = {
    "factors": (10, 100),
    "regularization": (1e-3, 1.0),
    "alpha": (1.0, 40.0),
    "n_iter": (5, 60),
}

# Maps hpo_metric name → Optuna study direction.
_METRIC_DIRECTION: dict[HPOMetric, str] = {
    "loss": "minimize",
    "precision": "maximize",
    "recall": "maximize",
    "ndcg": "maximize",
}


class TrialConfig(BaseModel):
    """An Optuna-sampled configuration passed to one mapped training step."""

    idx: int
    params: Hyperparameters
    metric: HPOMetric


class TrialResult(TrialConfig):
    """The metric artifact emitted by one mapped HPO training step."""

    value: float


def _get_metric_value(state: EpochState, hpo_metric: str) -> float:
    """Extract the HPO objective value from an epoch state."""
    mapping: dict[str, float] = {
        "loss": state.loss,
        "precision": state.precision_at_k,
        "recall": state.recall_at_k,
        "ndcg": state.ndcg_at_k,
    }
    if hpo_metric not in mapping:
        raise ValueError(f"Unknown hpo_metric: {hpo_metric!r}. Choose from {list(mapping)}")
    return mapping[hpo_metric]


def _train_als_subsample(
    train_pd: pd.DataFrame,
    val_pd: pd.DataFrame,
    factors: int,
    regularization: float,
    alpha: float,
    n_iter: int,
    n_workers: int,
    recommender_cls: type[BaseRecommender],
    hpo_metric: str = "loss",
) -> float:
    """
    Train on a subsample and return the final value of hpo_metric for this trial.
    """

    _, _, states = recommender_cls.train(
        train_data=train_pd,
        val_data=val_pd,
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        n_iter=n_iter,
        n_workers=n_workers,
        seed=42,
        eval_every_n_epochs=1,
    )

    return _get_metric_value(states[-1], hpo_metric)


# NOTE: The suggest_hpo_trials function is not a ZenML step; it runs in-memory Optuna sampling.
def suggest_hpo_trials(
    hpo_n_trials: int,
    hpo_metric: HPOMetric = "loss",
    seed: int = 42,
) -> Annotated[list[TrialConfig], "hpo_trial_configs"]:
    """Sample all trial configurations from an in-memory Optuna study."""
    if hpo_n_trials < 1:
        raise ValueError("hpo_n_trials must be at least 1")
    if hpo_metric not in _METRIC_DIRECTION:
        raise ValueError(
            f"Unknown hpo_metric: {hpo_metric!r}. Choose from {list(_METRIC_DIRECTION)}"
        )

    study = optuna.create_study(
        direction=_METRIC_DIRECTION[hpo_metric],
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    trial_configs: list[TrialConfig] = []
    for trial_idx in range(hpo_n_trials):
        trial = study.ask()
        trial_configs.append(
            TrialConfig(
                idx=trial_idx,
                params=Hyperparameters(
                    factors=trial.suggest_int(
                        "factors", HPO_SPACES["factors"][0], HPO_SPACES["factors"][1]
                    ),
                    regularization=trial.suggest_float(
                        "regularization",
                        HPO_SPACES["regularization"][0],
                        HPO_SPACES["regularization"][1],
                        log=True,
                    ),
                    alpha=float(
                        trial.suggest_int(
                            "alpha",
                            int(HPO_SPACES["alpha"][0]),
                            int(HPO_SPACES["alpha"][1]),
                        )
                    ),
                    n_iter=trial.suggest_int(
                        "n_iter", HPO_SPACES["n_iter"][0], HPO_SPACES["n_iter"][1]
                    ),
                ),
                metric=hpo_metric,
            )
        )

    return trial_configs


@step(
    enable_cache=False,
    experiment_tracker=(
        experiment_tracker.name if experiment_tracker else None
    ),  # Use custom experiment tracker if available
)
def run_hpo_trial(
    trial_config: TrialConfig,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    n_workers: int = 4,
    hpo_subsample_fraction: float = 0.2,
    recommender_class_name: str = "workflows.matrix_factorization.models.als_implicit_recommender.ALSImplicitRecommender",
    seed: int = 42,
) -> Annotated[TrialResult, "trial_result"]:
    """
    Run a single Optuna HPO trial. Multiple instances run in parallel via ZenML fan-out.

    Args:
        trial_config: Optuna-sampled configuration for this trial.
        train_data: Training ratings pandas DataFrame.
        val_data: Validation ratings pandas DataFrame.
        n_workers: Number of parallel partition workers.
        hpo_subsample_fraction: Fraction of training data to use for this trial.
        recommender_class_name: Fully-qualified class path of a BaseRecommender subclass to train.
        seed: Base random seed for reproducible trial subsampling.

    Returns:
        trial_result dict: {trial_idx, value, params}
    """
    recommender_cls: type[BaseRecommender] = load_recommender_class(recommender_class_name)

    train_pd = train_data
    val_pd = val_data
    if hpo_subsample_fraction < 1.0:
        trial_seed = seed + trial_config.idx
        train_pd = train_pd.sample(frac=hpo_subsample_fraction, random_state=trial_seed)
        val_pd = val_pd.sample(frac=hpo_subsample_fraction, random_state=trial_seed)

    logger.info(
        "Trial %d: %d training ratings, %d val ratings",
        trial_config.idx,
        len(train_pd),
        len(val_pd),
    )

    result_value = _train_als_subsample(
        train_pd,
        val_pd,
        trial_config.params.factors,
        trial_config.params.regularization,
        trial_config.params.alpha,
        trial_config.params.n_iter,
        n_workers,
        recommender_cls,
        hpo_metric=trial_config.metric,
    )

    logger.info(
        "Trial %d complete. %s=%.4f, params: %s",
        trial_config.idx,
        trial_config.metric,
        result_value,
        trial_config.params,
    )
    log_metadata(
        metadata={
            "hpo_trial_idx": trial_config.idx,
            "hpo_metric": trial_config.metric,
            "hpo_value": result_value,
            "hpo_params": trial_config.params.model_dump(),
        }
    )

    return TrialResult(
        idx=trial_config.idx,
        value=result_value,
        params=trial_config.params,
        metric=trial_config.metric,
    )


@step(enable_cache=False)
def collect_best_hpo_params(
    trial_results: list[TrialResult],
) -> Annotated[Hyperparameters, "best_hyperparams"]:
    """
    Fan-in: select the best result artifact produced by all mapped trial steps.

    Args:
        trial_results: Result artifacts from all mapped HPO trial steps.

    Returns:
        best_hyperparams: Hyperparameters dataclass with the best hyperparameters found across all trials.
    """
    if not trial_results:
        raise ValueError("No HPO trial results were received")
    metric = trial_results[0].metric
    if any(result.metric != metric for result in trial_results):
        raise ValueError("All HPO trial results must use the same metric")

    best_result = (
        min(trial_results, key=lambda result: result.value)
        if metric == "loss"
        else max(trial_results, key=lambda result: result.value)
    )
    logger.info(
        "HPO complete. Best params: %s (%s=%.4f) across %d trials",
        best_result.params,
        metric,
        best_result.value,
        len(trial_results),
    )

    log_metadata(
        metadata={
            "hpo_best_params": best_result.params.model_dump(),
            "hpo_best_metric": metric,
            "hpo_best_value": best_result.value,
            "hpo_n_trials": len(trial_results),
        }
    )

    return best_result.params
