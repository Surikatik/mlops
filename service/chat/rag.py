import threading
import time
from pathlib import Path

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.ollama import (
    OllamaDocumentEmbedder,
    OllamaTextEmbedder,
)
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

from .knowledge import build_documents

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_VERSION = "v2"
NO_ANSWER = "В базе знаний нет информации по этому вопросу."
MIN_RULE_LENGTH = 30
MAX_HISTORY = 6
MAX_QUESTION_LENGTH = 1000


def load_prompt(name):
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


class ChatEngine:
    def __init__(
        self,
        url="http://localhost:11434",
        chat_model="qwen2.5:1.5b",
        embedding_model="paraphrase-multilingual",
        top_k=5,
        temperature=0.1,
        min_score=0.38,
        root=".",
        timeout=300,
    ):
        self.url = url
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.temperature = temperature
        self.min_score = min_score
        self.root = root
        self.timeout = timeout
        self.system_prompt = load_prompt("system.txt")
        self.user_prompt = load_prompt("user.txt")
        self.store = None
        self.pipeline = None
        self.text_embedder = OllamaTextEmbedder(
            model=embedding_model, url=url, timeout=timeout
        )
        self.lock = threading.Lock()

    def index(self):
        store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        embedder = OllamaDocumentEmbedder(
            model=self.embedding_model,
            url=self.url,
            timeout=self.timeout,
            progress_bar=False,
        )
        documents = embedder.run(documents=build_documents(self.root))["documents"]
        store.write_documents(documents)
        return store

    def build_pipeline(self, store):
        pipeline = Pipeline()
        pipeline.add_component(
            "embedding_retriever",
            InMemoryEmbeddingRetriever(document_store=store, top_k=self.top_k),
        )
        pipeline.add_component(
            "bm25_retriever",
            InMemoryBM25Retriever(document_store=store, top_k=self.top_k),
        )
        pipeline.add_component(
            "joiner",
            DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=self.top_k),
        )
        pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(variables=["documents", "question"], required_variables="*"),
        )
        pipeline.add_component(
            "llm",
            OllamaChatGenerator(
                model=self.chat_model,
                url=self.url,
                timeout=self.timeout,
                generation_kwargs={"temperature": self.temperature, "num_predict": 400},
            ),
        )
        pipeline.connect("embedding_retriever", "joiner")
        pipeline.connect("bm25_retriever", "joiner")
        pipeline.connect("joiner", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "llm.messages")
        return pipeline

    def ensure_ready(self):
        with self.lock:
            if self.pipeline is None:
                self.store = self.index()
                self.pipeline = self.build_pipeline(self.store)

    def messages(self, history):
        messages = [ChatMessage.from_system(self.system_prompt)]
        for turn in (history or [])[-MAX_HISTORY:]:
            content = str(turn.get("content", ""))[:MAX_QUESTION_LENGTH]
            if turn.get("role") == "assistant":
                messages.append(ChatMessage.from_assistant(content))
            elif turn.get("role") == "user":
                messages.append(ChatMessage.from_user(content))
        messages.append(ChatMessage.from_user(self.user_prompt))
        return messages

    def guard(self, answer):
        rules = [
            line.strip().lower()
            for line in self.system_prompt.splitlines()
            if len(line.strip()) >= MIN_RULE_LENGTH
        ]
        lowered = answer.lower()
        if any(rule in lowered for rule in rules):
            return NO_ANSWER
        return answer

    def search_query(self, question, history):
        previous = [
            str(turn.get("content", ""))
            for turn in (history or [])
            if turn.get("role") == "user"
        ]
        if not previous:
            return question
        return f"{previous[-1][:MAX_QUESTION_LENGTH]} {question}"

    def ask(self, question, history=None):
        question = str(question).strip()[:MAX_QUESTION_LENGTH]
        if not question:
            raise ValueError("Пустой вопрос")
        self.ensure_ready()
        started = time.perf_counter()
        query = self.search_query(question, history)
        embedding = self.text_embedder.run(text=query)["embedding"]
        nearest = self.store.embedding_retrieval(embedding, top_k=1)
        if not nearest or nearest[0].score < self.min_score:
            return {
                "answer": NO_ANSWER,
                "sources": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "latency": time.perf_counter() - started,
            }
        result = self.pipeline.run(
            {
                "embedding_retriever": {"query_embedding": embedding},
                "bm25_retriever": {"query": query},
                "prompt_builder": {
                    "template": self.messages(history),
                    "question": question,
                },
            },
            include_outputs_from={"joiner"},
        )
        reply = result["llm"]["replies"][0]
        documents = result["joiner"]["documents"]
        meta = reply.meta or {}
        usage = meta.get("usage") or {}
        return {
            "answer": self.guard((reply.text or "").strip()),
            "sources": [
                {
                    "source": document.meta["source"],
                    "title": document.meta["title"],
                    "kind": document.meta["kind"],
                }
                for document in documents
            ],
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            },
            "latency": time.perf_counter() - started,
        }
