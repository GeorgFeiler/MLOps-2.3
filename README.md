# MLOps-2.3 - Управление потоком операций (xFlow)

## Краткий пошаговый обзор проделанной работы

### 1. Определение внешнего источника данных:

* В данном случае внешний источник данных - это YouTube API. Мы используем библиотеку pyyoutube для извлечения комментариев и количества лайков для видео по заданному запросу. Конкретно здесь применён запрос "skill factory инженерия машинного обучения".

### 2. Постановка задачи и выбор модели:

* Задача состоит в том, чтобы предсказать количество лайков для видео на YouTube. Для этого используется модель линейной регрессии.
    Метрика, используемая для оценки модели, - это коэффициент детерминации (R²).

### 3. Создание инфраструктуры:

* Установлено и настроено виртуальное окружение Python, а также установлены и настроены Apache Airflow и MLflow.

### 4. Создание Python скриптов:

* [get_data.py](https://github.com/GeorgFeiler/MLOps-2.3/blob/main/scripts/get_data.py): Извлечение данных с YouTube и сохранение суммарного количества лайков в файл.
* [process_data.py](https://github.com/GeorgFeiler/MLOps-2.3/blob/main/scripts/process_data.py): Нормализация данных и их сохранение в файл.
* [train_test_split_data.py](https://github.com/GeorgFeiler/MLOps-2.3/blob/main/scripts/train_test_split_data.py): Разделение данных на тренировочные и тестовые наборы.
* [train_model.py](https://github.com/GeorgFeiler/MLOps-2.3/blob/main/scripts/train_model.py): Обучение модели линейной регрессии на тренировочных данных и сохранение модели.
* [test_model.py](https://github.com/GeorgFeiler/MLOps-2.3/blob/main/scripts/test_model.py): Оценка качества модели на тестовых данных.

### 5. Добавление кода для Airflow:

* [youtube_comments_score.py](https://github.com/GeorgFeiler/MLOps-2.3/blob/main/airflow/dags/youtube_comments_score.py): Определение DAG для автоматического запуска всех описанных выше операций на регулярной основе.

### 6. Добавление кода для MLflow:

* В каждом из скриптов добавлены команды для логирования артефактов, метрик и скриптов в MLflow, что позволяет мониторить выполнение конвейера и анализировать результаты.

## Технологии

* Oracle VirtualBox 7.0.18
* Ubuntu 24.04 LTS
* Python 3.9.19
* AirFlow 2.9.3
* MLflow 2.14.2 (включая MLflow Tracking Server)
* Visual Studio Code 1.90.2
* YouTube API

## Библиотеки

[requirements.txt](https://github.com/GeorgFeiler/MLOps-2.3/blob/main/requirements.txt)

* aiohttp==3.9.5
* aiohttp-retry==2.8.3
* aiosignal==1.3.1
* apache-airflow==2.9.3
* attrs==23.2.0
* diskcache==5.6.3
* distro==1.9.0
* dulwich==0.22.1
* fsspec==2024.6.1
* funcy==2.0
* joblib==1.4.2
* mlflow==2.14.2
* numpy==1.21.6
* pandas==1.5.0
* pathspec==0.12.1
* pygit2==1.15.0
* python-dateutil==2.9.0.post0
* pytz==2024.1
* PyYAML==6.0
* pyyoutube==0.8.1 
* requests==2.32.3
* scikit-learn==1.1.3
* scipy==1.9.3
* tqdm==4.66.4
