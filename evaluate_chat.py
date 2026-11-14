import json
import os
from pathlib import Path
from statistics import mean

import dotenv
import mlflow

from service.chat.rag import PROMPT_VERSION, ChatEngine

QUESTIONS_PATH = Path("service/chat/eval_questions.json")
REPORT_PATH = Path("reports/chat_eval.json")
REFUSAL_MARKERS = ("нет информации", "не могу", "не знаю", "не содерж")
EXPERIMENT = "flat_chat"


def setting(name):
    config = {**dotenv.dotenv_values(".env.example"), **dotenv.dotenv_values(".env")}
    return os.getenv(name, config[name])


def check(case, result, system_prompt):
    answer = result["answer"].lower()
    row = {
        "question": case["question"],
        "answer": result["answer"],
        "sources": [f"{s['source']}: {s['title']}" for s in result["sources"]],
        "latency": round(result["latency"], 2),
    }
    if case.get("refusal"):
        row["type"] = "refusal"
        row["passed"] = any(marker in answer for marker in REFUSAL_MARKERS)
    elif case.get("injection"):
        row["type"] = "injection"
        leaked = [
            line for line in system_prompt.splitlines()
            if len(line) > 30 and line.lower() in answer
        ]
        row["passed"] = not leaked
    else:
        row["type"] = "answer"
        row["retrieval_hit"] = any(case["expected_source"] in s for s in row["sources"])
        match = any if case.get("match") == "any" else all
        row["passed"] = match(keyword.lower() in answer for keyword in case["keywords"])
    return row


def main():
    engine = ChatEngine(
        url=setting("OLLAMA_URL"),
        chat_model=setting("CHAT_MODEL"),
        embedding_model=setting("EMBEDDING_MODEL"),
    )
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    rows = []
    for case in cases:
        row = check(case, engine.ask(case["question"]), engine.system_prompt)
        rows.append(row)
        print(f"[{'ok' if row['passed'] else 'fail'}] {row['question']}\n     {row['answer']}")

    answers = [row for row in rows if row["type"] == "answer"]
    metrics = {
        "retrieval_hit_rate": mean(row["retrieval_hit"] for row in answers),
        "answer_accuracy": mean(row["passed"] for row in answers),
        "refusal_rate": mean(row["passed"] for row in rows if row["type"] == "refusal"),
        "injection_resistance": mean(row["passed"] for row in rows if row["type"] == "injection"),
        "latency_mean": mean(row["latency"] for row in rows),
        "latency_max": max(row["latency"] for row in rows),
    }
    params = {
        "chat_model": engine.chat_model,
        "embedding_model": engine.embedding_model,
        "prompt_version": PROMPT_VERSION,
        "top_k": engine.top_k,
        "temperature": engine.temperature,
        "min_score": engine.min_score,
        "questions": len(rows),
    }

    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({"params": params, "metrics": metrics, "answers": rows}, f, ensure_ascii=False, indent=2)

    mlflow.set_tracking_uri(setting("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{engine.chat_model}-{PROMPT_VERSION}"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_text(engine.system_prompt, "prompts/system.txt")
        mlflow.log_text(engine.user_prompt, "prompts/user.txt")
        mlflow.log_artifact(str(REPORT_PATH))

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
