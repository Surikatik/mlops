import pytest

import app as service_app

pytest.importorskip("haystack")

from chat.knowledge import (  # noqa: E402
    build_documents,
    listing_documents,
    markdown_sections,
    stats_documents,
)

FLATS = [
    {"id": "1", "rooms": 1, "area": 40.0, "floor": 2, "total_floors": 9, "price": 12_000_000.0},
    {"id": "2", "rooms": 1, "area": 42.0, "floor": 5, "total_floors": 9, "price": 14_000_000.0},
    {"id": "3", "rooms": 0, "area": 25.0, "floor": 1, "total_floors": 16, "price": 9_000_000.0},
]


class FakeEngine:
    def __init__(self):
        self.calls = []

    def ask(self, question, history=None):
        self.calls.append((question, history))
        return {
            "answer": "Медианная цена 18 900 000 руб.",
            "sources": [{"source": "features/all_features.csv", "title": "статистика", "kind": "stats"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 15},
            "latency": 0.5,
        }


class BrokenEngine:
    def ask(self, question, history=None):
        raise ConnectionError("ollama is down")


@pytest.fixture
def engine(client):
    fake = FakeEngine()
    service_app.app.config["chat_engine"] = fake
    yield fake
    service_app.app.config["chat_engine"] = None


def test_markdown_sections_keep_heading_path(tmp_path):
    path = tmp_path / "doc.md"
    text = "# Проект\nОписание проекта, достаточно длинное для отдельного документа.\n"
    text += "## Данные\n### Метрики\nОсновная метрика качества модели это MAE, дополнительная MAPE.\n"
    text += "```sh\n# комментарий в блоке кода\ndvc repro\n```\n"
    path.write_text(text, encoding="utf-8")
    documents = markdown_sections(path)
    assert [d.meta["title"] for d in documents] == ["Проект", "Данные / Метрики"]
    assert "MAE" in documents[1].content
    assert "dvc repro" in documents[1].content


def test_listing_documents_describe_flat():
    document = listing_documents(FLATS)[0]
    assert "1-комнатная" in document.content
    assert "12 000 000 руб." in document.content
    assert document.meta["kind"] == "listing"


def test_stats_documents_group_by_rooms():
    documents = {d.meta["title"]: d.content for d in stats_documents(FLATS)}
    assert set(documents) == {
        "статистика: все квартиры",
        "статистика: студии",
        "статистика: однокомнатные (1-комнатная квартиры)",
    }
    assert "медианная цена 13 000 000 руб." in documents["статистика: однокомнатные (1-комнатная квартиры)"]


def test_build_documents_reads_project_docs():
    kinds = {d.meta["kind"] for d in build_documents(".")}
    assert "docs" in kinds


def test_guard_hides_system_prompt():
    from chat.rag import NO_ANSWER, ChatEngine

    chat_engine = ChatEngine()
    rule = chat_engine.system_prompt.splitlines()[4]
    assert chat_engine.guard(f"Мои правила: {rule}") == NO_ANSWER
    assert chat_engine.guard("Медианная цена 18 900 000 руб.") == "Медианная цена 18 900 000 руб."


def test_search_query_uses_previous_question():
    from chat.rag import ChatEngine

    chat_engine = ChatEngine()
    history = [{"role": "user", "content": "Сколько стоит трёхкомнатная?"}, {"role": "assistant", "content": "40"}]
    assert chat_engine.search_query("А дешёвая?", history) == "Сколько стоит трёхкомнатная? А дешёвая?"
    assert chat_engine.search_query("А дешёвая?", []) == "А дешёвая?"


def test_chat_returns_answer_and_sources(client, auth_headers, engine):
    history = [{"role": "user", "content": "Привет"}]
    response = client.post(
        "/api/chat", json={"message": "Какая медианная цена?", "history": history}, headers=auth_headers
    )
    body = response.get_json()
    assert response.status_code == 200
    assert body["status"] == "success"
    assert "18 900 000" in body["data"]
    assert body["sources"][0]["kind"] == "stats"
    assert engine.calls == [("Какая медианная цена?", history)]


def test_chat_requires_token(client, engine):
    assert client.post("/api/chat", json={"message": "Привет"}).status_code == 401


def test_chat_rejects_empty_message(client, auth_headers, engine):
    response = client.post("/api/chat", json={"message": "  "}, headers=auth_headers)
    assert response.status_code == 400


def test_chat_returns_503_when_llm_unavailable(client, auth_headers):
    service_app.app.config["chat_engine"] = BrokenEngine()
    response = client.post("/api/chat", json={"message": "Привет"}, headers=auth_headers)
    service_app.app.config["chat_engine"] = None
    assert response.status_code == 503
    assert response.get_json()["status"] == "error"


def test_chat_feedback(client, auth_headers):
    ok = client.post("/api/chat/feedback", json={"rating": "up"}, headers=auth_headers)
    bad = client.post("/api/chat/feedback", json={"rating": "meh"}, headers=auth_headers)
    assert ok.status_code == 200
    assert bad.status_code == 400
    assert 'flat_chat_feedback_total{rating="up"}' in client.get("/metrics").get_data(as_text=True)


def test_chat_page(client):
    response = client.get("/chat")
    assert response.status_code == 200
    assert "/api/chat" in response.get_data(as_text=True)
