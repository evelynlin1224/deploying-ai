from services.semantic_search import KnowledgeBaseService, LocalSemanticEmbedding


def test_embedding_dimension():
    embedder = LocalSemanticEmbedding()
    vector = embedder.embed_query("semantic Chroma search")
    assert len(vector) == 96


def test_semantic_search_returns_hits():
    service = KnowledgeBaseService()
    hits = service.search("How does Chroma persistence work?", n_results=2)
    assert len(hits) == 2
    assert any("Chroma" in hit.title or "Chroma" in hit.text for hit in hits)
