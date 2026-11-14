from preprocess import extract_features_from_text, extract_id_from_url


def test_extract_features_from_card_text():
    text = "2-комн. квартира | 54,3 м² | 7/16 эт. | 18 500 000 ₽"
    assert extract_features_from_text(text) == {
        "rooms": 2,
        "area": 54.3,
        "floor": 7,
        "total_floors": 16,
        "price": 18_500_000,
    }


def test_extract_features_studio():
    assert extract_features_from_text("Студия | 25 м² | 3/9 эт.")["rooms"] == 0


def test_extract_features_from_empty_text():
    assert set(extract_features_from_text("").values()) == {None}


def test_extract_id_from_url():
    url = "https://domclick.ru/card/sale__flat__2079790812"
    assert extract_id_from_url(url) == "2079790812"
    assert extract_id_from_url("https://domclick.ru/") is None
