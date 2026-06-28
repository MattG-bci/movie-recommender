from datetime import datetime

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"
PIPELINE_CMD = f"cd {PROJECT_DIR} && PYTHONPATH={PROJECT_DIR}/src python3 src/main.py"

with DAG(
    dag_id="ingestion_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ingestion"],
) as dag:
    ingest_users = BashOperator(
        task_id="ingest_users",
        bash_command=f"{PIPELINE_CMD} ingest-users",
    )

    ingest_movies = BashOperator(
        task_id="ingest_movies",
        bash_command=f"{PIPELINE_CMD} ingest-movies-command",
    )

    ingest_ratings = BashOperator(
        task_id="ingest_ratings",
        bash_command=f"{PIPELINE_CMD} ingest-ratings",
    )

    [ingest_users, ingest_movies] >> ingest_ratings
