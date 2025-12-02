from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Добавляем путь к проекту
sys.path.append(os.getenv("PROJECT_PATH") + "/airflow/scripts")

from update_dataset import update_dataset

with DAG(
    dag_id="update_dataset_monthly",
    description="Download latest candles and update dataset",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@monthly",
    catchup=False,
    tags=["dataset", "tradebot"]
):
    update_task = PythonOperator(
        task_id="update_dataset",
        python_callable=update_dataset
    )

    update_task
