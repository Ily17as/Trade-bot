# airflow/dags/monthly_pipeline_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import sys

project_path = os.getenv("PROJECT_PATH", "/opt/airflow/Trade-bot")
scripts_path = os.path.join(project_path, "airflow", "scripts")
sys.path.append(scripts_path)

from update_dataset import update_dataset
from train_ml import train_ml
from train_cv import train_cv
from train_rl import train_rl

with DAG(
    dag_id="monthly_training_pipeline",
    description="Update dataset and retrain models monthly",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@monthly",
    catchup=False,
    tags=["tradebot", "training"],
) as dag:

    update_task = PythonOperator(
        task_id="update_dataset",
        python_callable=update_dataset,
    )

    ml_task = PythonOperator(
        task_id="train_ml",
        python_callable=train_ml,
    )

    cv_task = PythonOperator(
        task_id="train_cv",
        python_callable=train_cv,
    )

    rl_task = PythonOperator(
        task_id="train_rl",
        python_callable=train_rl,
    )

    # Порядок выполнения:
    update_task >> ml_task >> cv_task >> rl_task
