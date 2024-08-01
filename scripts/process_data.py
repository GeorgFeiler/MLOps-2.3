import pandas as pd
import mlflow

# Установка URI для подключения к серверу MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Установка эксперимента с именем "process_data" в MLflow
mlflow.set_experiment("process_data")

with mlflow.start_run():
    # Чтение данных из файла
    df = pd.read_csv('/opt/clearml/projects/MLOps-2.3/datasets/data.csv', header=None)
    
    # Нормализация данных
    df[0] = (df[0] - df[0].min()) / (df[0].max() - df[0].min())

    # Сохранение предобработанных данных в новый файл
    with open('/opt/clearml/projects/MLOps-2.3/datasets/data_processed.csv', 'w') as f:
        for i, item in enumerate(df[0].values):
            f.write("{},{}\n".format(i, item))
    
    # Логирование артефактов в MLflow
    mlflow.log_artifact('/opt/clearml/projects/MLOps-2.3/datasets/data_processed.csv')
    mlflow.log_artifact('/opt/clearml/projects/MLOps-2.3/scripts/process_data.py')
    mlflow.end_run()
