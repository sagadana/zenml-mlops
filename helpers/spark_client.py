"""
helpers/spark_client.py

Shared Spark/Hive helpers used across workflow steps.
"""

from __future__ import annotations

import os
import re

import pandas as pd

_SPARK_IDENTITY = "root"
_JAVA_USER_NAME_OPTION = f"-Duser.name={_SPARK_IDENTITY}"


_HIVE_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$"
)


def validate_hive_identifier(identifier: str) -> None:
    """Validate a Hive table identifier (optionally qualified with one database)."""
    if not _HIVE_IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(
            "identifier must be an unquoted table name optionally qualified with one database."
        )


def query_hive(query: str, app_name: str, spark_master_url: str) -> pd.DataFrame:
    """Execute a Hive SQL query through Spark and return the result as a pandas DataFrame."""
    _ensure_spark_identity()

    from pyspark.sql import SparkSession

    with (
        SparkSession.builder.appName(app_name)
        .master(spark_master_url)
        .config("spark.sql.catalogImplementation", "hive")
        .enableHiveSupport()
        .getOrCreate()
    ) as session:
        return session.sql(query).toPandas()


def _ensure_spark_identity() -> None:
    """Set a fallback Unix identity before Spark starts Hadoop login."""
    for variable_name in ("USER", "LOGNAME", "HADOOP_USER_NAME", "SPARK_USER"):
        os.environ.setdefault(variable_name, _SPARK_IDENTITY)
    for variable_name in ("JAVA_TOOL_OPTIONS", "HADOOP_OPTS"):
        current_value = os.environ.get(variable_name, "")
        if _JAVA_USER_NAME_OPTION not in current_value.split():
            os.environ[variable_name] = (
                f"{current_value} {_JAVA_USER_NAME_OPTION}".strip()
            )
