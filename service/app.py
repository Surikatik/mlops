import argparse
import os
from logging.config import dictConfig

import dotenv
import joblib
from flask import Flask, render_template, request
from flask_cors import CORS
from flask_httpauth import HTTPTokenAuth

try:
    from . import monitoring
    from . import predict as prediction_models
except ImportError:
    import monitoring
    import predict as prediction_models

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "service/flask.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)

app = Flask(__name__)

config = {
    **dotenv.dotenv_values(".env.example"),
    **{k: v for k, v in dotenv.dotenv_values(".env").items() if v is not None},
}
config.update({k: os.environ[k] for k in list(config) if k in os.environ})

cors = CORS(
    app,
    resources={r"/api/*": {"origins": config.get("ORIGIN", "http://localhost:5000")}},
)

auth = HTTPTokenAuth(scheme="Bearer")
tokens = {config["APP_TOKEN"]: "user1"}


@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]


# Модель по умолчанию задаётся в .env
MODEL_NAME = config.get("MODEL_NAME", "baseline_const")


def load_predictor(name):
    """Возвращает predict(X) по имени модели из predict.py, URI модели MLflow или пути к .pkl."""
    if name in prediction_models.MODELS:
        return prediction_models.MODELS[name]
    if name.startswith(("models:/", "runs:/")):
        import mlflow

        mlflow.set_tracking_uri(
            config.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        )
        return mlflow.sklearn.load_model(name).predict
    return joblib.load(name).predict


app.config["predict"] = load_predictor(MODEL_NAME)
drift_monitor = monitoring.init_monitoring(app)


# Маршрут для отображения формы
@app.route("/")
def index():
    return render_template("index.html")


# Маршрут для обработки данных формы
@app.route("/api/numbers", methods=["POST"])
@auth.login_required
def process_numbers():

    data = request.get_json()

    app.logger.info(f"Requst data: {data}")
    try:
        row = prediction_models.build_features(
            data["area"], data["rooms"], data["floor"], data["total_floors"]
        )
    except (KeyError, TypeError, ValueError):
        return {"status": "error", "data": "Ошибка парсинга данных"}, 400

    try:
        price = int(app.config["predict"]([row])[0])
    except NotImplementedError as exc:
        app.logger.error(f"Model error: {exc}")
        return {"status": "error", "data": str(exc)}, 501

    drift_monitor.observe(
        {
            "area": data["area"],
            "rooms": data["rooms"],
            "floor": data["floor"],
            "total_floors": data["total_floors"],
        }
    )
    monitoring.PREDICTED_PRICE.observe(price)
    app.logger.info(f"Predicted price: {price}")

    return {"status": "success", "data": price}


if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Имя модели из service/predict.py, URI модели MLflow или путь к .pkl",
        default=MODEL_NAME,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=5000)
    args = parser.parse_args()

    app.config["predict"] = load_predictor(args.model)
    app.logger.info(f"Use model: {args.model}")
    app.run(host=args.host, port=args.port, debug=True)
