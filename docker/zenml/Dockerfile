# syntax=docker/dockerfile:1
# NOTE: Client's Zenml version must match the server's version.

FROM zenmldocker/zenml-server:0.96.2

USER root

RUN pip install --no-cache-dir pymysql

# Create a dedicated non-root runtime user for the pipeline
ENV HOME=/home/app
RUN groupadd --system app && \
    useradd --system --gid app --create-home --home-dir ${HOME} app && \
    mkdir -p ${HOME}/.config && \
    chown -R app:app ${HOME}

# Set the user to the non-root user
USER app

# Set the PATH environment variable to include the user's local bin directory
ENV PATH="${HOME}/.local/bin:${PATH}"

EXPOSE 8080
