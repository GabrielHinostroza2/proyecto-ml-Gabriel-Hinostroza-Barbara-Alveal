from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'gabriel_hinostroza',
    'retries': 1,
    'start_date': datetime(2025, 11, 1)
}

with DAG(
    dag_id='proyecto_ml_kedro_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='DAG para ejecutar el pipeline de Kedro del proyecto ML'
) as dag:

    # Tarea 1: Preparar entorno
    preparar_entorno = BashOperator(
        task_id='preparar_entorno',
        bash_command='cd /opt/airflow && pip install -r requirements.txt'
    )

    # Tarea 2: Ejecutar pipeline de Kedro
    ejecutar_pipeline = BashOperator(
        task_id='ejecutar_pipeline',
        bash_command='cd /opt/airflow && kedro run'
    )

    preparar_entorno >> ejecutar_pipeline
