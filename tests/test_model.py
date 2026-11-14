import pytest

from predict import MODELS_DIR, build_features, predict

TRAINED = [
    name for name in ("lin_reg", "ridge", "random_forest")
    if (MODELS_DIR / f"{name}.pkl").exists()
]

TOLERANCE = 0.02

pytestmark = pytest.mark.skipif(not TRAINED, reason="модели не обучены, выполните dvc repro")


@pytest.mark.parametrize("name", TRAINED)
def test_price_is_plausible(name):
    price = predict(name, [build_features(45, 2, 5, 16)])[0]
    assert 3_000_000 < price < 100_000_000


@pytest.mark.parametrize("name", TRAINED)
def test_price_does_not_fall_when_area_grows(name):
    areas = [30, 40, 50, 60, 70]
    prices = predict(name, [build_features(area, 2, 5, 16) for area in areas])
    assert prices[-1] > prices[0]
    assert all(b >= a * (1 - TOLERANCE) for a, b in zip(prices, prices[1:]))


@pytest.mark.parametrize("name", TRAINED)
def test_prediction_is_deterministic(name):
    X = [build_features(45, 2, 5, 16)]
    assert predict(name, X)[0] == predict(name, X)[0]
