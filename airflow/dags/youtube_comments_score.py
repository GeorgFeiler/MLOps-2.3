from airflow import DAG
from airflow.operators.bash import BashOperator
import datetime as dt

args = {
    "owner": "gera",
    "start_date": dt.datetime(2024, 7, 30),
    "retries": 5,
    "retry_delay": dt.timedelta(seconds=10),
    "depends_on_past": False
}

with DAG(
    dag_id='skill_factory_youtube_comments_score',
    default_args=args,
    schedule_interval='*/1 * * * *',
    tags=['skill factory youtube', 'MLOps-2.3', 'score'],
) as dag:
    
    # Определение задач в DAG
    get_data = BashOperator(task_id='get_data', bash_command="python3 /opt/clearml/projects/MLOps-2.3/scripts/get_data.py", dag=dag)
    process_data = BashOperator(task_id='process_data', bash_command="python3 /opt/clearml/projects/MLOps-2.3/scripts/process_data.py", dag=dag)
    train_test_split_data = BashOperator(task_id='train_test_split_data', bash_command="python3 /opt/clearml/projects/MLOps-2.3/scripts/train_test_split_data.py", dag=dag)
    train_model = BashOperator(task_id='train_model', bash_command="python3 /opt/clearml/projects/MLOps-2.3/scripts/train_model.py", dag=dag)
    test_model = BashOperator(task_id='test_model', bash_command="python3 /opt/clearml/projects/MLOps-2.3/scripts/test_model.py", dag=dag)
    
    # Определение последовательности выполнения задач
    get_data >> process_data >> train_test_split_data >> train_model >> test_model
