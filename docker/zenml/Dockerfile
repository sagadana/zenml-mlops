# syntax=docker/dockerfile:1
FROM zenmldocker/zenml-server:0.94.6

USER root

RUN pip install --no-cache-dir \
    zenml[mysql] \
    pymysql

RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /home/app app && \
    mkdir -p /home/app/.config && \
    chown -R app:app /home/app

USER app

EXPOSE 8080
