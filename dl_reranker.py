from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(query, retrieved_docs):
    pairs = [(query, doc["text"]) for doc in retrieved_docs]
    scores = reranker.predict(pairs)

    # attach scores and sort
    for doc, score in zip(retrieved_docs, scores):
        doc["rerank_score"] = float(score)

    reranked = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)
    return reranked
