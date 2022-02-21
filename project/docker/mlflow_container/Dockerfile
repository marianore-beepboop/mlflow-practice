FROM continuumio/miniconda3:latest

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN echo "export LC_ALL=$LC_ALL" >> /etc/profile.d/locale.sh
RUN echo "export LANG=$LANG" >> /etc/profile.d/locale.sh
ENV GUNICORN_CMD_ARGS="--timeout 500"

# Copy files
COPY ./entrypoints/start_mlflow.sh ./entrypoints/start_mlflow.sh
RUN chmod +x ./entrypoints/start_mlflow.sh

# Set up SSH
RUN apt-get update && apt-get install -y openssh-server
RUN useradd -m -s /bin/bash dockeruser
RUN mkdir /var/run/sshd
RUN echo 'dockeruser:123' | chpasswd

# Set up permissions
RUN chown dockeruser:dockeruser /opt/conda/
RUN mkdir -p /home/dockeruser/mlruns /home/dockeruser/mlruns/.trash
RUN chown 777 -R /home/dockeruser/mlruns/

# Install system-wide packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev

# Install needed packages
RUN pip install -U pip && \
    pip install boto3 psycopg2 mlflow>=1.9.1 && \
    pip install gunicorn>=19.9.0 \
    pip install jupyter \
    pip install scikit-learn
