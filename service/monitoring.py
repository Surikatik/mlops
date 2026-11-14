import json
import math
import threading
import time
from collections import deque
from pathlib import Path

from flask import Response, g, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REFERENCE_PATH = (
    Path(__file__).resolve().parent.parent / "reports" / "feature_reference.json"
)
WINDOW_SIZE = 200
MIN_SAMPLES = 30
EPSILON = 1e-4

REQUESTS = Counter(
    "flat_requests_total",
    "Количество HTTP-запросов",
    ["endpoint", "method", "status"],
)
LATENCY = Histogram(
    "flat_request_duration_seconds",
    "Время обработки HTTP-запроса",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
PREDICTED_PRICE = Histogram(
    "flat_predicted_price_rub",
    "Распределение предсказанной цены",
    buckets=(5e6, 1e7, 1.5e7, 2e7, 2.5e7, 3e7, 4e7, 5e7, 7.5e7, 1e8),
)
FEATURE_MEAN = Gauge(
    "flat_feature_mean",
    "Среднее значение признака в скользящем окне запросов",
    ["feature"],
)
FEATURE_REFERENCE_MEAN = Gauge(
    "flat_feature_reference_mean",
    "Среднее значение признака в обучающей выборке",
    ["feature"],
)
FEATURE_PSI = Gauge(
    "flat_feature_psi",
    "Population Stability Index признака относительно обучающей выборки",
    ["feature"],
)
WINDOW_SAMPLES = Gauge(
    "flat_drift_window_samples",
    "Количество запросов в скользящем окне",
)
CHAT_REQUESTS = Counter(
    "flat_chat_requests_total",
    "Количество запросов к чату",
    ["status"],
)
CHAT_LATENCY = Histogram(
    "flat_chat_duration_seconds",
    "Время ответа RAG-пайплайна",
    buckets=(1, 2.5, 5, 10, 20, 30, 60, 120, 300),
)
CHAT_TOKENS = Counter(
    "flat_chat_tokens_total",
    "Количество токенов языковой модели",
    ["type"],
)
CHAT_FEEDBACK = Counter(
    "flat_chat_feedback_total",
    "Оценки ответов чата пользователями",
    ["rating"],
)


def psi(expected, actual):
    value = 0.0
    for e, a in zip(expected, actual):
        e = max(e, EPSILON)
        a = max(a, EPSILON)
        value += (a - e) * math.log(a / e)
    return value


def bin_fractions(values, edges):
    counts = [0] * (len(edges) + 1)
    for value in values:
        index = sum(value > edge for edge in edges)
        counts[index] += 1
    total = len(values)
    return [count / total for count in counts]


class DriftMonitor:
    def __init__(self, reference_path=REFERENCE_PATH, window_size=WINDOW_SIZE):
        self.reference = {}
        if Path(reference_path).exists():
            with open(reference_path, encoding="utf-8") as f:
                self.reference = json.load(f)
        self.windows = {name: deque(maxlen=window_size) for name in self.reference}
        self.lock = threading.Lock()
        for name, ref in self.reference.items():
            FEATURE_REFERENCE_MEAN.labels(feature=name).set(ref["mean"])

    def observe(self, values):
        with self.lock:
            for name, window in self.windows.items():
                if name not in values:
                    continue
                window.append(float(values[name]))
                FEATURE_MEAN.labels(feature=name).set(sum(window) / len(window))
                if len(window) >= MIN_SAMPLES:
                    ref = self.reference[name]
                    actual = bin_fractions(window, ref["edges"])
                    FEATURE_PSI.labels(feature=name).set(
                        psi(ref["fractions"], actual)
                    )
            if self.windows:
                WINDOW_SAMPLES.set(len(next(iter(self.windows.values()))))


def init_monitoring(app):
    monitor = DriftMonitor()
    app.extensions["drift_monitor"] = monitor

    @app.before_request
    def start_timer():
        g.start_time = time.perf_counter()

    @app.after_request
    def record_request(response):
        if request.path == "/metrics":
            return response
        endpoint = request.url_rule.rule if request.url_rule else "unknown"
        REQUESTS.labels(
            endpoint=endpoint, method=request.method, status=response.status_code
        ).inc()
        if hasattr(g, "start_time"):
            LATENCY.labels(endpoint=endpoint).observe(
                time.perf_counter() - g.start_time
            )
        return response

    @app.route("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    return monitor
