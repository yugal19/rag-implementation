from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any

nltk.download("punkt")

model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en")


def chunk_and_embed(text: str, num_chunks: int = 5) -> List[Dict[str, Any]]:
    sentences = sent_tokenize(text)

    if not sentences:
        return []

    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)

    kmeans = KMeans(n_clusters=min(num_chunks, len(sentences)), random_state=42)
    labels = kmeans.fit_predict(sentence_embeddings)

    # Group sentences into clusters
    clusters = {}
    for sent, label in zip(sentences, labels):
        clusters.setdefault(label, []).append(sent)

    # Merge sentences into chunks
    chunks = [" ".join(sents) for sents in clusters.values()]

    # Embed chunks
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True)

    # Prepare results
    results = [
        {"text": chunk, "embedding": chunk_embeddings[i]}
        for i, chunk in enumerate(chunks)
    ]
    return results
