# Курс MLOps
Это учебный проект, направленный на создание модели машинного обучения для прогнозирования цен на жилье. 
Папка с [лекциями курса](https://drive.google.com/drive/folders/1HsI7IYoc-BrBrA6SRhxm1tzyCLtwnKht?usp=sharing).  
Модель использует различные характеристики объектов недвижимости для предсказания их рыночной стоимости.

## Защита семинаров

Отчет по семинарам сдается в личной беседе.  
Сделайте форк репозитория или заведите свой. 
Под каждую тему заводите отдельную ветку, merge в main через pull request. 
Ветки после PR не удалять! 
После прохождения курса ваш репо помимо main должен содержать следующие dev ветки: 
1. Разработка дизайн документа
2. Внедрение DVC
3. Внедрение MLflow
4. Внедрение дашборда мониторинга, CI/CD github actions
5. Внедрение чата Haystack + Ollama


### Архитектура сервиса 
![](img/arch.png)


### Как запустить
1. Клонируйте репозиторий:
```bash
git clone https://github.com/labintsev/mlops.git
```

2. Создайте venv и установите зависимости:
```bash
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

4. Запустите сервис (модель по умолчанию берётся из `MODEL_NAME` в `.env`)
```sh
python service/app.py
# или с явным указанием модели из service/predict.py
python service/app.py -m baseline_const
```
Веб приложение доступно по ссылке `http://127.0.0.1:5000`
API endpoint доступен  по ссылке `http://127.0.0.1:5000/api/numbers`

Чтобы добавить новую модель, объявите функцию `def my_model(X): ...` в `service/predict.py`
и зарегистрируйте её в словаре `MODELS` — после этого её можно выбрать через `-m` / `MODEL_NAME`.

## Сбор данных
Для сбора данных используем [Instant data scraper](https://chromewebstore.google.com/detail/instant-data-scraper/ofaokhiedipichpaobibbnahnkdoiiah?pli=1)


### Признаки
Используемые данные включают следующие характеристики:
* Площадь жилья
* Количество комнат
* Этаж квартиры
* Этажей в доме


## Версионирование данных (DVC)
Сырые данные (`data/*.xlsx`) и результаты обработки не хранятся в Git: в репозитории лежат только указатели `data.dvc` и `dvc.lock`, а сами файлы находятся в DVC remote.

Remote по умолчанию `myremote` указывает на локальную папку `../dvcstore` рядом с репозиторием. Чтобы использовать другое хранилище:
```sh
dvc remote add -d -f myremote /path/to/dvcstore
```

Получить данные и воспроизвести конвейер:
```sh
dvc pull
dvc repro
dvc metrics show
```

Стадии конвейера описаны в `dvc.yaml`:

| Стадия | Команда | Вход | Выход |
| :--- | :--- | :--- | :--- |
| `preprocess` | `python preprocess.py` | `data/` | `features/all_features.csv`, `reports/data_stats.json` |
| `train` | `python train.py` | `features/all_features.csv`, `params.yaml` | `models/*.pkl`, `reports/metrics.json` |

Добавить новые данные:
```sh
dvc add data
dvc repro
dvc push
git add data.dvc dvc.lock reports/data_stats.json
git commit -m "update data"
```
Вернуться к предыдущей версии данных: `git checkout <commit> data.dvc dvc.lock && dvc checkout`.

## Обучение и трекинг экспериментов (MLflow)
Скрипт `train.py` обучает модели из `params.yaml` и константный baseline (медиана цены). Схема валидации: отложенная выборка 20% и 5-fold кросс-валидация на обучающей части.

Каждая модель записывается в MLflow отдельным запуском эксперимента `flat_price`:
* параметры модели и разбиения;
* метрики MAE и MAPE на train, validation (среднее по фолдам) и holdout;
* артефакт модели с сигнатурой и примером входа;
* теги `git_commit` и `data_version` (хэш данных из `data.dvc`).

Модели сохраняются в `models/<имя>.pkl` и регистрируются в реестре под именем `flat_price`. Лучшая по MAE на кросс-валидации модель получает алиас `champion`, если на отложенной выборке она точнее baseline.

```sh
dvc repro
dvc metrics show
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Интерфейс MLflow доступен по ссылке `http://127.0.0.1:5000`, поэтому сервис при одновременном запуске нужно поднять на другом порту или запустить MLflow с `--port 5001`.

Адрес трекинга задаётся переменной `MLFLOW_TRACKING_URI`, по умолчанию `sqlite:///mlflow.db`.

Запуск сервиса с обученной моделью:
```sh
python service/app.py -m ridge
python service/app.py -m models:/flat_price@champion
python service/app.py -m models/random_forest.pkl
```

Результаты на текущей версии данных (80 объявлений):

| Модель | MAE val, руб. | MAE holdout, руб. | MAPE holdout |
| :--- | ---: | ---: | ---: |
| `baseline_const` | 6 664 107 | 6 307 500 | 24.8% |
| `lin_reg` | 2 969 554 | 2 999 843 | 13.3% |
| `ridge` | 2 884 349 | 2 491 780 | 10.6% |
| `random_forest` | 3 097 604 | 2 864 231 | 11.7% |

## Содержание курса

### Тема 1. Введение в MLOps
* Введение в MLOps
* Бизнес анализ
* Дизайн документ

### Тема 2 Data Ops
* Форматы и модели данных
* Настройка конвеера сбора данных
* Отбор и разметка данных. Несбалансированные выборки
* Аугментация данных
* Предварительная обработка  признаков

### Тема 3 Model Ops

* Разделение выборки и утечка данных
* Постановка задачи ML
* Виды моделей ML и рекомендации по выбору
* Выбор функции потерь
* Ансамбли моделей
* Разработка, трекинг и логирование
* Mlflow: сценарии, архитектура, альтернативы
* Auto ML
* Распределенное обучение
* Офлайн оценка
* Оценка подмножеств (срезов) данных

### Тема 4 Dev Ops
* Разработка моделей предиктивной аналитики и оценка метрик
* Внедрение моделей в бизнес
* Инфраструктура инференса моделей


### Тема 5 LLM Ops

* Модель, промпты и данные
* RAG и оценка качества
* Развертывание и мониторинг

## Лицензирование
Этот проект распространяется под лицензией MIT. 
Смотрите файл LICENSE для деталей.

## Контакты
Для вопросов и предложений обращайтесь:
* Email: ailabintsev@fa.ru
