from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import mlflow

# Установка URI для подключения к серверу MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Установка эксперимента с именем "test_model" в MLflow
mlflow.set_experiment("test_model")

with mlflow.start_run():
    # Чтение тестовых данных
    df = pd.read_csv('/opt/clearml/projects/MLOps-2.3/datasets/data_test.csv', header=None)
    df.columns = ['id', 'counts']
    
    # Логирование информации о данных
    print(f"Данные загружены: {df.head()}")
    print(f"Количество NaN значений в 'counts': {df['counts'].isna().sum()}")
    
    # Обработка NaN значений
    df['counts'] = df['counts'].fillna(0)
    
    # Логирование информации после обработки NaN
    print(f"Количество NaN значений в 'counts' после обработки: {df['counts'].isna().sum()}")
    
    # Подготовка данных для оценки модели
    X = df['id'].values.reshape(-1, 1)
    y = df['counts'].values
    
    # Загрузка обученной модели
    with open('/opt/clearml/projects/MLOps-2.3/models/data.pickle', 'rb') as f:
        model = pickle.load(f)
    
    # Оценка модели
    score = model.score(X, y)
    print("score=", score)
    
    # Логирование метрики оценки модели и скрипта в MLflow
    mlflow.log_metric("score", score)
    mlflow.log_artifact('/opt/clearml/projects/MLOps-2.3/scripts/test_model.py')
    mlflow.end_run()
