from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import mlflow

# Установка URI для подключения к серверу MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Установка эксперимента с именем "train_model" в MLflow
mlflow.set_experiment("train_model")

# Чтение обучающих данных
df = pd.read_csv('/opt/clearml/projects/MLOps-2.3/datasets/data_train.csv', header=None)
df.columns = ['id', 'counts']

# Логирование информации о данных
print(f"Данные загружены: {df.head()}")
print(f"Количество NaN значений в 'counts': {df['counts'].isna().sum()}")

# Обработка NaN значений
df['counts'] = df['counts'].fillna(0)

# Логирование информации после обработки NaN
print(f"Количество NaN значений в 'counts' после обработки: {df['counts'].isna().sum()}")

# Подготовка данных для модели
X = df['id'].values.reshape(-1, 1)
y = df['counts'].values

# Инициализация и тренировка модели
model = LinearRegression()

with mlflow.start_run():
    # Логирование модели в MLflow
    mlflow.sklearn.log_model(model, artifact_path="lr", registered_model_name="lr")
    mlflow.log_artifact(local_path="/opt/clearml/projects/MLOps-2.3/scripts/train_model.py", artifact_path="train_model code")
    mlflow.end_run()

model.fit(X, y)

# Сохранение модели в файл
with open('/opt/clearml/projects/MLOps-2.3/models/data.pickle', 'wb') as f:
    pickle.dump(model, f)

print("Модель успешно обучена и сохранена.")
