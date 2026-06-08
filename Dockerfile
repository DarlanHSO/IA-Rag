FROM python:3.9-slim

RUN pip install mlflow==2.6.0 psycopg2-binary boto3 pymysql

RUN mkdir -p /mlflow
WORKDIR /mlflow

COPY start_mlflow.sh /mlflow/start_mlflow.sh
RUN sed -i 's/\r//' /mlflow/start_mlflow.sh && chmod +x /mlflow/start_mlflow.sh

EXPOSE 3000

CMD ["/bin/bash", "/mlflow/start_mlflow.sh"]