FROM daskdev/dask:2024.12.1

COPY docker/dask/start-scheduler.sh /opt/dask/start-scheduler.sh
COPY docker/dask/start-worker.sh /opt/dask/start-worker.sh
RUN chmod +x /opt/dask/start-scheduler.sh /opt/dask/start-worker.sh

EXPOSE 8786 8787
