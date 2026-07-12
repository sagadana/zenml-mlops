from enum import StrEnum

CFG_MODEL_NAME = "als_movie_recommender"
CFG_MODEL_DESCRIPTION = "ALS movie recommender model"
CFG_MODEL_ARTIFACT_NAME = "als_movie_model"
CFG_MODEL_PICKLE_FILENAME = "als_movie_recommender.pkl"

CFG_INFERENCE_LOGS_EXT = ".jsonl"

CFG_TRAINING_PIPELINE_NAME = "matrix_factorization_training"
CFG_TRAINING_PIPELINE_SNAPSHOT_NAME = "matrix_factorization_training_snapshot"
CFG_TRAINING_PIPELINE_SNAPSHOT_DESCRIPTION = (
    "Snapshot of the ALS training pipeline for matrix factorization."
)

CFG_SERVING_PIPELINE_NAME = "matrix_factorization_serving"
CFG_SERVING_PIPELINE_SNAPSHOT_NAME = "matrix_factorization_serving_snapshot"
CFG_SERVING_PIPELINE_SNAPSHOT_DESCRIPTION = (
    "Snapshot of the ALS serving pipeline for matrix factorization."
)

CFG_MONITORING_PIPELINE_NAME = "matrix_factorization_monitoring"
CFG_MONITORING_PIPELINE_SNAPSHOT_NAME = "matrix_factorization_monitoring_snapshot"
CFG_MONITORING_PIPELINE_SNAPSHOT_DESCRIPTION = (
    "Snapshot of the ALS monitoring pipeline for matrix factorization."
)


class CFG_DATASET_FIELD_NAMES(StrEnum):
    USER_ID = "userId"
    ITEM_ID = "movieId"
    RATING = "rating"
    TIMESTAMP = "timestamp"


class CFG_DATASET_FIELD_TYPES(StrEnum):
    USER_ID = "int32"
    ITEM_ID = "int32"
    RATING = "float32"
    TIMESTAMP = "int64"


class CFG_FEATURES_FIELD_NAMES(StrEnum):
    USER_ID = "user_idx"
    ITEM_ID = "item_idx"
    RATING = "rating"
    TIMESTAMP = "timestamp"


class CFG_PREDICTION_FIELD_NAMES(StrEnum):
    USER_ID = "user_id"
    ITEM_ID = "item_id"
    SCORE = "score"


class CFG_BATCH_PREDICTION_FIELD_NAMES(StrEnum):
    USER_ID = "user_id"
    RECOMMENDATIONS = "recommendations"


class CFG_RECS_FIELD_NAMES(StrEnum):
    RECORD_ID = "id"
    USER_ID = "userId"
    RECS = "recs"
    REC_ITEM_ID = "itemId"
    REC_SCORE = "score"
    REC_RANK = "rank"
    VERSION = "version"
    UPDATED_AT = "updated_at"


class CFG_RECS_LOG_FIELD_NAMES(StrEnum):
    USER_ID = "user_id"
    COUNT = "count"
    TOP_K = "top_k"
    LATENCY_MS = "latency_ms"
    TIMESTAMP = "timestamp"


__all__ = [
    "CFG_MODEL_NAME",
    "CFG_MODEL_ARTIFACT_NAME",
    "CFG_MODEL_PICKLE_FILENAME",
    "CFG_INFERENCE_LOGS_EXT",
    "CFG_DATASET_FIELD_NAMES",
    "CFG_DATASET_FIELD_TYPES",
    "CFG_FEATURES_FIELD_NAMES",
    "CFG_PREDICTION_FIELD_NAMES",
    "CFG_BATCH_PREDICTION_FIELD_NAMES",
    "CFG_RECS_FIELD_NAMES",
    "CFG_RECS_LOG_FIELD_NAMES",
]
