"""Service 2: semantic query over a small project knowledge base.

The primary implementation uses a ChromaDB PersistentClient.  A deterministic
local embedding function and precomputed document embeddings are included so the
repository does not depend on a separate embedding-generation job.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CSV_PATH = DATA_DIR / "knowledge_base.csv"
PRECOMPUTED_PATH = DATA_DIR / "precomputed_embeddings.json"
CHROMA_PATH = DATA_DIR / "chroma_db"
COLLECTION_NAME = "lumi_assignment_knowledge"

TOKEN_RE = re.compile(r"[a-z0-9_#+.\-/]+|[\u4e00-\u9fff]+", re.IGNORECASE)

# Concept boosts give the tiny local vectorizer a little semantic awareness.
CONCEPTS: dict[str, list[str]] = {
    "api_service": ["api", "http", "request", "weather", "forecast", "open-meteo", "json", "backend", "接口", "天气"],
    "semantic_search": ["semantic", "search", "retrieval", "vector", "embedding", "chroma", "chromadb", "knowledge", "query", "语义", "检索", "向量", "知识库"],
    "function_calling": ["function", "tool", "tool_call", "schema", "planner", "schedule", "plan", "函数", "工具", "计划"],
    "gradio_ui": ["gradio", "chat", "chatbot", "interface", "ui", "blocks", "textbox", "界面", "聊天"],
    "memory": ["memory", "history", "conversation", "context", "summary", "short-term", "turns", "记忆", "历史", "上下文"],
    "guardrails": ["guardrail", "restricted", "refuse", "system prompt", "prompt injection", "policy", "blocked", "安全", "限制", "系统提示"],
    "submission": ["github", "branch", "pull request", "pr", "readme", "submit", "submission", "repository", "repo", "分支", "提交"],
    "testing": ["test", "tests", "unittest", "debug", "validate", "pytest", "testing", "测试", "调试"],
    "data_limits": ["40 mb", "file size", "dataset", "csv", "sqlite", "pandas", "persistence", "文件", "数据"],
    "personality": ["personality", "persona", "tone", "lumi", "friendly", "ta", "角色", "语气"],
}


@dataclass
class SearchHit:
    doc_id: str
    title: str
    category: str
    text: str
    score: float


class LocalSemanticEmbedding:
    """Small deterministic embedding function suitable for a tiny demo dataset.

    This is not a replacement for high-quality neural embeddings.  It is designed
    for assignments where the vector store must be small, reproducible, and easy
    to share in GitHub.
    """

    def __init__(self, dim: int = 96):
        self.dim = dim
        self._concept_slots = self._assign_concept_slots()

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        vector = [0.0] * self.dim
        lowered = text.lower()

        # Hashed token features.
        for token in tokens:
            self._add_hash_feature(vector, token, weight=1.0)
            # Character n-grams help with plurals and nearby spellings.
            if len(token) >= 5 and not self._looks_cjk(token):
                for i in range(len(token) - 2):
                    self._add_hash_feature(vector, token[i : i + 3], weight=0.25)

        # Concept features approximate synonym handling for the assignment domain.
        for concept, keywords in CONCEPTS.items():
            boost = 0.0
            for keyword in keywords:
                if keyword in lowered:
                    boost += 1.5
            if boost:
                vector[self._concept_slots[concept]] += boost

        return self._normalize(vector)

    def _assign_concept_slots(self) -> dict[str, int]:
        slots: dict[str, int] = {}
        for index, concept in enumerate(CONCEPTS):
            slots[concept] = index % self.dim
        return slots

    def _tokenize(self, text: str) -> list[str]:
        return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]

    @staticmethod
    def _looks_cjk(token: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in token)

    def _add_hash_feature(self, vector: list[float], token: str, weight: float) -> None:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign * weight

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [round(value / norm, 8) for value in vector]


class KnowledgeBaseService:
    """Semantic query service backed by ChromaDB with a safe local fallback."""

    def __init__(
        self,
        csv_path: Path = CSV_PATH,
        precomputed_path: Path = PRECOMPUTED_PATH,
        chroma_path: Path = CHROMA_PATH,
    ) -> None:
        self.csv_path = csv_path
        self.precomputed_path = precomputed_path
        self.chroma_path = chroma_path
        self.embedder = LocalSemanticEmbedding()
        self.rows = self._load_rows()
        self.precomputed = self._load_or_compute_embeddings()
        self.collection = None
        self.chroma_status = "not initialized"
        self._init_chroma()

    def answer(self, question: str, n_results: int = 3) -> str:
        cleaned_question = re.sub(r"^/(kb|search|semantic)\s*", "", question.strip(), flags=re.IGNORECASE)
        if not cleaned_question:
            return "Ask me a question about the assignment requirements or this project’s design, and I’ll search the local Chroma knowledge base."

        hits = self.search(cleaned_question, n_results=n_results)
        if not hits:
            return "I searched the local knowledge base but did not find a confident match. Try mentioning a service, Chroma, Gradio, memory, guardrails, or submission."

        lead = "I searched the project knowledge base"
        if self.collection is not None:
            lead += " with ChromaDB persistence"
        else:
            lead += " with the local vector fallback because ChromaDB is not installed in this environment"
        lead += ". Here are the most relevant notes:\n\n"

        bullets = []
        for hit in hits:
            bullets.append(
                f"**{hit.title}** ({hit.category}, score {hit.score:.2f}) — {self._shorten(hit.text, 360)}"
            )
        synthesis = self._synthesize(cleaned_question, hits)
        return lead + "\n\n".join(bullets) + "\n\n" + synthesis

    def search(self, question: str, n_results: int = 3) -> list[SearchHit]:
        n_results = max(1, min(n_results, 5))
        query_embedding = self.embedder.embed_query(question)
        if self.collection is not None:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                )
                return self._hits_from_chroma(results)
            except Exception as exc:  # pragma: no cover - fallback path depends on installed Chroma version
                self.chroma_status = f"Chroma query failed; fallback active: {exc}"
        return self._fallback_search(query_embedding, n_results=n_results)

    def _load_rows(self) -> list[dict[str, str]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Knowledge base CSV not found: {self.csv_path}")
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    def _load_or_compute_embeddings(self) -> dict[str, list[float]]:
        if self.precomputed_path.exists():
            with self.precomputed_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            items = payload.get("items", [])
            mapping = {item["id"]: item["embedding"] for item in items}
            if all(row["id"] in mapping for row in self.rows):
                return mapping
        docs = [self._doc_text(row) for row in self.rows]
        embeddings = self.embedder.embed_documents(docs)
        return {row["id"]: embedding for row, embedding in zip(self.rows, embeddings)}

    def _init_chroma(self) -> None:
        try:
            import chromadb  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional course environment
            self.chroma_status = f"ChromaDB unavailable; using vector fallback: {exc}"
            return

        self.chroma_path.mkdir(parents=True, exist_ok=True)
        try:
            client = chromadb.PersistentClient(path=str(self.chroma_path))
            self.collection = client.get_or_create_collection(name=COLLECTION_NAME)
            if self.collection.count() == 0:
                self.collection.add(
                    ids=[row["id"] for row in self.rows],
                    documents=[self._doc_text(row) for row in self.rows],
                    metadatas=[{"title": row["title"], "category": row["category"]} for row in self.rows],
                    embeddings=[self.precomputed[row["id"]] for row in self.rows],
                )
            self.chroma_status = f"ChromaDB PersistentClient active at {self.chroma_path}"
        except Exception as exc:  # pragma: no cover - depends on Chroma runtime
            self.collection = None
            self.chroma_status = f"ChromaDB initialization failed; using vector fallback: {exc}"

    def _fallback_search(self, query_embedding: list[float], n_results: int) -> list[SearchHit]:
        scored: list[SearchHit] = []
        for row in self.rows:
            doc_embedding = self.precomputed[row["id"]]
            similarity = self._cosine(query_embedding, doc_embedding)
            scored.append(
                SearchHit(
                    doc_id=row["id"],
                    title=row["title"],
                    category=row["category"],
                    text=row["text"],
                    score=max(0.0, similarity),
                )
            )
        return sorted(scored, key=lambda hit: hit.score, reverse=True)[:n_results]

    def _hits_from_chroma(self, results: dict[str, Any]) -> list[SearchHit]:
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]
        hits: list[SearchHit] = []
        for doc_id, doc, metadata, distance in zip(ids, documents, metadatas, distances):
            # Chroma distances vary by index metric; convert to an intuitive similarity-ish score.
            score = 1.0 / (1.0 + float(distance)) if distance is not None else 0.0
            hits.append(
                SearchHit(
                    doc_id=str(doc_id),
                    title=str(metadata.get("title", doc_id)) if metadata else str(doc_id),
                    category=str(metadata.get("category", "knowledge")) if metadata else "knowledge",
                    text=str(doc),
                    score=score,
                )
            )
        return hits

    def _synthesize(self, question: str, hits: list[SearchHit]) -> str:
        top_categories = ", ".join(dict.fromkeys(hit.category for hit in hits))
        return (
            f"**Lumi’s take:** your question maps mostly to **{top_categories}**. "
            "Use these notes as grounding, then test the related code path in the Gradio app before submitting."
        )

    def _doc_text(self, row: dict[str, str]) -> str:
        return f"{row['title']}\nCategory: {row['category']}\n{row['text']}"

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 1].rstrip() + "…"
