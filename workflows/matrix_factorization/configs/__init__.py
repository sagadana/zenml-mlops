import os

CFG_MODEL_NAME = "als_movie_recommender"

CFG_DASK_SCHEDULER_ADDRESS = os.environ.get("DASK_SCHEDULER_ADDRESS", None)

__all__ = [
    "CFG_MODEL_NAME",
    "CFG_DASK_SCHEDULER_ADDRESS",
]
