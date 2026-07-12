"""Rebuild the local embeddings JSON and, when chromadb is installed, the Chroma store.

You do not need to run this for normal grading because data/precomputed_embeddings.json
is already included. Run it only after editing data/knowledge_base.csv.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from services.semantic_search import (  # noqa: E402
    CHROMA_PATH,
    COLLECTION_NAME,
    CSV_PATH,
    PRECOMPUTED_PATH,
    LocalSemanticEmbedding,
)


def main() -> None:
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    embedder = LocalSemanticEmbedding()
    docs = [f"{row['title']}\nCategory: {row['category']}\n{row['text']}" for row in rows]
    embeddings = embedder.embed_documents(docs)

    payload = {
        "embedding_model": "local_hash_semantic_v1",
        "dim": embedder.dim,
        "description": "Deterministic local semantic hash embeddings for the Assignment 2 demo knowledge base.",
        "items": [{"id": row["id"], "embedding": embedding} for row, embedding in zip(rows, embeddings)],
    }
    PRECOMPUTED_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {PRECOMPUTED_PATH}")

    try:
        import chromadb  # type: ignore
    except Exception as exc:
        print(f"chromadb is not installed, so only the JSON embeddings were rebuilt: {exc}")
        return

    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    collection.add(
        ids=[row["id"] for row in rows],
        documents=docs,
        metadatas=[{"title": row["title"], "category": row["category"]} for row in rows],
        embeddings=embeddings,
    )
    print(f"Seeded Chroma collection {COLLECTION_NAME!r} at {CHROMA_PATH}")


if __name__ == "__main__":
    main()
