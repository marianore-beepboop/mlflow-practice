FROM conda/miniconda3:latest

RUN mkdir -p /mlflow/mlruns

WORKDIR /mlflow

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN echo "export LC_ALL=$LC_ALL" >> /etc/profile.d/locale.sh
RUN echo "export LANG=$LANG" >> /etc/profile.d/locale.sh
ENV GUNICORN_CMD_ARGS="--timeout 500"

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev

RUN pip install -U pip && \
    # pip install --ignore-installed google-cloud-storage && \
    pip install boto3 psycopg2 mlflow==1.9.1 && \
    pip install gunicorn==19.9.0

COPY ./entrypoints/start.sh ./entrypoints/start.sh
RUN chmod +x ./entrypoints/start.sh

COPY ./entrypoints/start_ui.sh ./entrypoints/start_ui.sh
RUN chmod +x ./entrypoints/start_ui.sh

# EXPOSE 80
# EXPOSE 5000

CMD ["./entrypoints/start.sh"]
