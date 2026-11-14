import argparse
import os
import threading
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

chat_lock = threading.Lock()


def get_chat_engine():
    with chat_lock:
        if app.config.get("chat_engine") is None:
            try:
                from .chat.rag import ChatEngine
            except ImportError:
                from chat.rag import ChatEngine

            app.config["chat_engine"] = ChatEngine(
                url=config.get("OLLAMA_URL", "http://localhost:11434"),
                chat_model=config.get("CHAT_MODEL", "qwen2.5:1.5b"),
                embedding_model=config.get("EMBEDDING_MODEL", "paraphrase-multilingual"),
            )
        return app.config["chat_engine"]


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


@app.route("/chat")
def chat_page():
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
@auth.login_required
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message")
    history = data.get("history") or []
    if not isinstance(message, str) or not message.strip() or not isinstance(history, list):
        return {"status": "error", "data": "Ошибка парсинга данных"}, 400

    try:
        result = get_chat_engine().ask(message, [t for t in history if isinstance(t, dict)])
    except Exception as exc:
        app.logger.error(f"Chat error: {exc}")
        monitoring.CHAT_REQUESTS.labels(status="error").inc()
        return {"status": "error", "data": "Языковая модель недоступна"}, 503

    monitoring.CHAT_REQUESTS.labels(status="success").inc()
    monitoring.CHAT_LATENCY.observe(result["latency"])
    monitoring.CHAT_TOKENS.labels(type="prompt").inc(result["usage"]["prompt_tokens"])
    monitoring.CHAT_TOKENS.labels(type="completion").inc(result["usage"]["completion_tokens"])
    app.logger.info(f"Chat question: {message!r}, answer: {result['answer']!r}")

    return {"status": "success", "data": result["answer"], "sources": result["sources"]}


@app.route("/api/chat/feedback", methods=["POST"])
@auth.login_required
def chat_feedback():
    data = request.get_json(silent=True) or {}
    rating = data.get("rating")
    if rating not in ("up", "down"):
        return {"status": "error", "data": "Ошибка парсинга данных"}, 400
    monitoring.CHAT_FEEDBACK.labels(rating=rating).inc()
    app.logger.info(f"Chat feedback: {rating}, question: {data.get('question')!r}")
    return {"status": "success", "data": rating}


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
