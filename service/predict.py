"""Модели предсказания стоимости квартиры.

Каждая модель — функция, которая принимает матрицу признаков X
(список строк в порядке FEATURES) и возвращает последовательность цен.

Чтобы добавить новую модель, достаточно объявить функцию и зарегистрировать
её в MODELS — она сразу станет доступна через `--model` и `MODEL_NAME`.
"""

from functools import lru_cache
from pathlib import Path

import joblib

FEATURES = (
    "total_meters",
    "floors_count",
    "rooms_1",
    "rooms_2",
    "rooms_3",
    "first_floor",
    "last_floor",
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def build_features(area, rooms, floor, total_floors):
    area = float(area)
    rooms = int(rooms)
    floor = int(floor)
    total_floors = int(total_floors)
    features = {
        "total_meters": area,
        "floors_count": total_floors,
        "rooms_1": int(rooms == 1),
        "rooms_2": int(rooms == 2),
        "rooms_3": int(rooms == 3),
        "first_floor": int(floor == 1),
        "last_floor": int(floor == total_floors),
    }
    return [features[name] for name in FEATURES]


@lru_cache(maxsize=None)
def load_trained(name):
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise NotImplementedError(f"{name} ещё не обучена")
    return joblib.load(path)


def baseline_const(X):
    try:
        return load_trained("baseline_const").predict(X)
    except NotImplementedError:
        return [10_000_000 for _ in X]


def lin_reg(X):
    return load_trained("lin_reg").predict(X)


def ridge(X):
    return load_trained("ridge").predict(X)


def random_forest(X):
    return load_trained("random_forest").predict(X)


MODELS = {
    "baseline_const": baseline_const,
    "lin_reg": lin_reg,
    "ridge": ridge,
    "random_forest": random_forest,
}


def predict(model_name, X):
    return MODELS[model_name](X)
