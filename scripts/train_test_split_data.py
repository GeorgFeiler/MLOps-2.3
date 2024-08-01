import pandas as pd
import numpy as np
import mlflow

# Установка URI для подключения к серверу MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Установка эксперимента с именем "train_test_split_data" в MLflow
mlflow.set_experiment("train_test_split_data")

with mlflow.start_run():
    # Чтение нормализованных данных
    df = pd.read_csv('/opt/clearml/projects/MLOps-2.3/datasets/data_processed.csv', header=None)
    
    # Разделение данных на обучающие и тестовые наборы
    idxs = np.array(df.index.values)
    np.random.shuffle(idxs)
    l = int(len(df) * 0.7)
    train_idxs = idxs[:l]
    test_idxs = idxs[l + 1:]
    
    df.loc[train_idxs, :].to_csv('/opt/clearml/projects/MLOps-2.3/datasets/data_train.csv', header=None, index=None)
    df.loc[test_idxs, :].to_csv('/opt/clearml/projects/MLOps-2.3/datasets/data_test.csv', header=None, index=None)
    
    # Логирование артефактов разбиения данных и скрипта в MLflow
    mlflow.log_artifact('/opt/clearml/projects/MLOps-2.3/datasets/data_train.csv')
    mlflow.log_artifact('/opt/clearml/projects/MLOps-2.3/datasets/data_test.csv')
    mlflow.log_artifact('/opt/clearml/projects/MLOps-2.3/scripts/train_test_split_data.py')
    mlflow.end_run()
