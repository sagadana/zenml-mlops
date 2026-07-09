FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    boto3==1.35.90 \
    mlflow==2.19.0 \
    PyMySQL==1.1.1

WORKDIR /mlflow

COPY docker/mlflow/start.sh /opt/mlflow/start.sh
RUN chmod +x /opt/mlflow/start.sh

EXPOSE 5000

CMD ["/opt/mlflow/start.sh"]
