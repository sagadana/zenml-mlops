FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir evidently==0.7.21

WORKDIR /app

COPY docker/evidently/start.sh /opt/evidently/start.sh
RUN chmod +x /opt/evidently/start.sh

EXPOSE 8000

CMD ["/opt/evidently/start.sh"]
