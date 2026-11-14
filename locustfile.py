import csv
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

TOKEN = os.getenv("APP_TOKEN", "change_me")
AREA_SHIFT = float(os.getenv("AREA_SHIFT", "0"))
FEATURES_PATH = Path("features/all_features.csv")
FALLBACK = [
    {"area": 25.0, "rooms": 0, "floor": 3, "total_floors": 9},
    {"area": 40.2, "rooms": 1, "floor": 18, "total_floors": 33},
    {"area": 55.0, "rooms": 2, "floor": 8, "total_floors": 16},
    {"area": 76.6, "rooms": 3, "floor": 11, "total_floors": 16},
]


def load_flats():
    if not FEATURES_PATH.exists():
        return FALLBACK
    with open(FEATURES_PATH, encoding="utf-8-sig") as f:
        return [
            {
                "area": float(row["area"]),
                "rooms": int(float(row["rooms"])),
                "floor": int(float(row["floor"])),
                "total_floors": int(float(row["total_floors"])),
            }
            for row in csv.DictReader(f)
        ]


FLATS = load_flats()


class FlatUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        flat = dict(random.choice(FLATS))
        flat["area"] = round(flat["area"] + random.gauss(0, 1) + AREA_SHIFT, 1)
        self.client.post(
            "/api/numbers", json=flat, headers={"Authorization": f"Bearer {TOKEN}"}
        )

    @task(1)
    def index(self):
        self.client.get("/")
