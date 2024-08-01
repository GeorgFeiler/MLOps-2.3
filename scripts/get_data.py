import requests
import json
from pyyoutube import Api
import os
import mlflow
from mlflow.tracking import MlflowClient

# Установка переменных окружения для MLflow
os.environ["MLFLOW_REGISTRY_URI"] = "/opt/clearml/projects/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_data")

# API ключ для доступа к YouTube Data API
key = "AIzaSyCp7Jxz_F39BnoHLHfRrPA11uHWDLN0d5I"
api = Api(api_key=key)

# Поиск видео по ключевым словам
query = "'skill factory инженерия машинного обучения'"
video = api.search_by_keywords(q=query, search_type=["video"], count=20, limit=50)

maxResults = 100
nextPageToken = ""
s = 0

with mlflow.start_run():
    # Извлечение количества лайков для каждого видео
    for i, id_ in enumerate([x.id.videoId for x in video.items]):
        uri = "https://www.googleapis.com/youtube/v3/commentThreads?" + \
              "key={}&textFormat=plainText&" + \
              "part=snippet&" + \
              "videoId={}&" + \
              "maxResults={}&" + \
              "pageToken={}"
        uri = uri.format(key, id_, maxResults, nextPageToken)
        content = requests.get(uri).text
        data = json.loads(content)
        for item in data['items']:
            s += int(item['snippet']['topLevelComment']['snippet']['likeCount'])
    # Логирование скрипта в MLflow
    mlflow.log_artifact(local_path="/opt/clearml/projects/MLOps-2.3/scripts/get_data.py", artifact_path="get_data code")
    mlflow.end_run()

# Сохранение суммарного количества лайков в файл
with open('/opt/clearml/projects/MLOps-2.3/datasets/data.csv', 'a') as f:
    f.write("{}\n".format(s))
