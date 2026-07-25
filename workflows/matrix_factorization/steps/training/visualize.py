"""
steps/training/visualize.py

ZenML step: visualize_training

Renders an interactive HTML report of ALS training metrics over epochs using
Plotly Express. Produces a single HTMLString artifact that can be viewed in
the ZenML dashboard.

Metrics visualized:
  - Training loss
  - RMSE
  - Precision@K
  - Recall@K
  - NDCG@K
"""

from __future__ import annotations

import logging
from typing import Annotated

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from zenml import step
from zenml.types import HTMLString

from workflows.matrix_factorization.models.base_recommender import EpochStates

logger = logging.getLogger(__name__)

_TEMPLATE = "plotly_white"


@step
def visualize_training(
    training_states: EpochStates,
) -> Annotated[HTMLString, "training_visualization"]:
    """Render an interactive HTML report of training metrics over epochs.

    Args:
        training_states: Per-epoch training states from train_als.

    Returns:
        HTMLString artifact with embedded Plotly charts viewable in ZenML dashboard.
    """
    records = [
        {
            "epoch": s.epoch,
            "Loss": s.loss,
            "RMSE": s.rmse,
            f"Precision@{s.k}": s.precision_at_k,
            f"Recall@{s.k}": s.recall_at_k,
            f"NDCG@{s.k}": s.ndcg_at_k,
            "Elapsed Time (s)": s.elapsed_time,
            "CPU (%)": s.cpu_percent,
            "Memory (MiB)": s.memory_mb,
            "GPU Memory (MiB)": s.gpu_memory_mb,
        }
        for s in training_states
    ]
    df = pd.DataFrame(records)

    k = training_states[0].k if len(training_states) > 0 else "K"
    n_epochs = len(df)

    # ── Build individual charts ───────────────────────────────────────────────
    fig_loss = px.line(
        df,
        x="epoch",
        y="Loss",
        title="Loss over Epochs",
        markers=True,
        template=_TEMPLATE,
    )
    fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Loss", height=400)

    metrics_source_title = (
        "Training" if training_states[-1].metrics_source == "train" else "Validation"
    )

    fig_rmse = px.line(
        df,
        x="epoch",
        y="RMSE",
        title="RMSE over Epochs",
        markers=True,
        template=_TEMPLATE,
        color_discrete_sequence=["#EF553B"],
    )
    fig_rmse.update_layout(xaxis_title="Epoch", yaxis_title="RMSE", height=400)

    ranking_cols = [f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}"]
    df_ranking = df[["epoch"] + ranking_cols].melt(
        id_vars="epoch", var_name="Metric", value_name="Value"
    )
    fig_ranking = px.line(
        df_ranking,
        x="epoch",
        y="Value",
        color="Metric",
        title=f"Ranking Metrics @{k} over Epochs",
        markers=True,
        template=_TEMPLATE,
    )
    fig_ranking.update_layout(xaxis_title="Epoch", yaxis_title="Score", height=400)

    fig_time = px.bar(
        df,
        x="epoch",
        y="Elapsed Time (s)",
        title="Elapsed Time per Epoch",
        template=_TEMPLATE,
        color="Elapsed Time (s)",
        color_continuous_scale="Blues",
    )
    fig_time.update_layout(xaxis_title="Epoch", yaxis_title="Elapsed Time (s)", height=400)

    # ── Resource charts ───────────────────────────────────────────────────────
    fig_cpu = px.line(
        df,
        x="epoch",
        y="CPU (%)",
        title="CPU Utilisation per Epoch",
        markers=True,
        template=_TEMPLATE,
        color_discrete_sequence=["#00CC96"],
    )
    fig_cpu.update_layout(xaxis_title="Epoch", yaxis_title="CPU (%)", height=400)

    fig_mem = px.line(
        df,
        x="epoch",
        y="Memory (MiB)",
        title="Process RSS Memory per Epoch",
        markers=True,
        template=_TEMPLATE,
        color_discrete_sequence=["#AB63FA"],
    )
    fig_mem.update_layout(xaxis_title="Epoch", yaxis_title="Memory (MiB)", height=400)

    has_gpu = df["GPU Memory (MiB)"].notna().any()
    fig_gpu = None
    if has_gpu:
        fig_gpu = px.line(
            df,
            x="epoch",
            y="GPU Memory (MiB)",
            title="GPU VRAM per Epoch",
            markers=True,
            template=_TEMPLATE,
            color_discrete_sequence=["#FFA15A"],
        )
        fig_gpu.update_layout(xaxis_title="Epoch", yaxis_title="GPU Memory (MiB)", height=400)

    # ── Summary stats ─────────────────────────────────────────────────────────
    final = df.iloc[-1]
    best_ndcg_epoch = df.loc[df[f"NDCG@{k}"].idxmax(), "epoch"]
    best_ndcg = df[f"NDCG@{k}"].max()

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html_parts: list[str] = []
    html_parts.append("""<!DOCTYPE html>
    <html>
        <head>
        <meta charset="utf-8">
            <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    background: #f8f9fa; margin: 0; padding: 24px; color: #333; }
            h1   { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }
            h2   { color: #34495e; margin-top: 32px; }
            .summary { display: flex; flex-wrap: wrap; gap: 16px; margin: 24px 0; }
            .card    { background: #fff; border-radius: 8px; padding: 16px 24px;
                        box-shadow: 0 1px 4px rgba(0,0,0,.12); min-width: 160px; flex: 1; }
            .card h3 { margin: 0 0 6px; font-size: 13px; color: #888; text-transform: uppercase; }
            .card p  { margin: 0; font-size: 26px; font-weight: 700; color: #2c3e50; }
            .chart   { background: #fff; border-radius: 8px; padding: 16px;
                        box-shadow: 0 1px 4px rgba(0,0,0,.12); margin-bottom: 24px; }
            </style>
        </head>
        <body>
            <h1>ALS Training Validation Report</h1>
        
    """)

    html_parts.append(f"""
    <div class="summary">
        <div class="card"><h3>Epochs</h3><p>{n_epochs}</p></div>
        <div class="card"><h3>Final Loss</h3><p>{final["Loss"]:.3f}</p></div>
        <div class="card"><h3>Final RMSE</h3><p>{final["RMSE"]:.3f}</p></div>
        <div class="card"><h3>Best NDCG@{k}</h3><p>{best_ndcg:.3f} (ep {best_ndcg_epoch})</p></div>
        <div class="card"><h3>Precision@{k}</h3><p>{final[f"Precision@{k}"]:.3f}</p></div>
        <div class="card"><h3>Recall@{k}</h3><p>{final[f"Recall@{k}"]:.3f}</p></div>
        <div class="card"><h3>Avg CPU (%)</h3><p>{df["CPU (%)"].mean():.1f}</p></div>
        <div class="card"><h3>Peak Memory (MiB)</h3><p>{df["Memory (MiB)"].max():.0f}</p></div>
    </div>
    """)

    charts: list[tuple[str, go.Figure]] = [
        ("Training Loss", fig_loss),
        (f"{metrics_source_title} RMSE", fig_rmse),
        (f"{metrics_source_title} Ranking Metrics @{k}", fig_ranking),
        ("Elapsed Time per Epoch", fig_time),
        ("CPU Utilisation per Epoch", fig_cpu),
        ("Process RSS Memory per Epoch", fig_mem),
    ]
    if fig_gpu is not None:
        charts.append(("GPU VRAM per Epoch", fig_gpu))

    for title, fig in charts:
        chart_html = fig.to_html(full_html=False, include_plotlyjs=True)
        html_parts.append(f"""
        <div class="chart">
            <h2>{title}</h2>
            {chart_html}
        </div>
        """)

    html_parts.append("</body></html>")
    return HTMLString("".join(html_parts))
