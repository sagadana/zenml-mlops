FROM python:3.12-slim

USER root

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir boto3 mlflow pymysql

WORKDIR /mlflow

# Create a dedicated non-root runtime user.
RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /home/app app

COPY docker/mlflow/start.sh /opt/mlflow/start.sh
RUN chmod +x /opt/mlflow/start.sh
RUN chown -R app:app /mlflow /opt/mlflow /home/app

USER app

EXPOSE 5000

CMD ["/opt/mlflow/start.sh"]
