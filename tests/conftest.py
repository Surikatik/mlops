import pytest

import app as service_app

TOKEN = service_app.config["APP_TOKEN"]


@pytest.fixture
def client():
    service_app.app.config["TESTING"] = True
    service_app.app.config["predict"] = service_app.load_predictor("baseline_const")
    return service_app.app.test_client()


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def flat():
    return {"area": 45.5, "rooms": 2, "total_floors": 16, "floor": 7}
