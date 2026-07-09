# syntax=docker/dockerfile:1
FROM zenmldocker/zenml-server:0.94.6

RUN pip install --no-cache-dir \
    zenml[mysql] \
    pymysql

EXPOSE 8080
