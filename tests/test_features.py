import pytest

from predict import FEATURES, MODELS, build_features, predict


def test_build_features_order_and_values():
    row = build_features(area=45.5, rooms=2, floor=7, total_floors=16)
    assert dict(zip(FEATURES, row)) == {
        "total_meters": 45.5,
        "floors_count": 16,
        "rooms_1": 0,
        "rooms_2": 1,
        "rooms_3": 0,
        "first_floor": 0,
        "last_floor": 0,
    }


def test_build_features_first_and_last_floor():
    first = dict(zip(FEATURES, build_features(30, 1, 1, 9)))
    last = dict(zip(FEATURES, build_features(30, 1, 9, 9)))
    assert (first["first_floor"], first["last_floor"]) == (1, 0)
    assert (last["first_floor"], last["last_floor"]) == (0, 1)


def test_build_features_studio_has_no_room_flags():
    row = dict(zip(FEATURES, build_features(25, 0, 3, 9)))
    assert row["rooms_1"] == row["rooms_2"] == row["rooms_3"] == 0


def test_build_features_rejects_text():
    with pytest.raises(ValueError):
        build_features("abc", 1, 1, 9)


def test_baseline_predicts_for_every_row():
    X = [build_features(30, 1, 2, 9), build_features(80, 3, 5, 16)]
    prices = predict("baseline_const", X)
    assert len(prices) == 2
    assert all(price > 0 for price in prices)


def test_registered_models():
    assert {"baseline_const", "lin_reg"} <= set(MODELS)
