from transformers import pipeline

classifier = pipeline(
    "text-classification", model="distilbert-base-uncased-finetuned-sst-2-english"
)


def classify_chunks(chunks: list[str]):
    """
    Classify sentiment of each chunk and return as list of dicts.
    """
    results = classifier(chunks)
    tagged_chunks = []
    for chunk, result in zip(chunks, results):
        tagged_chunks.append(
            {
                "text": chunk,
                "sentiment": result["label"],
                "score": round(result["score"], 3),
            }
        )
    return tagged_chunks


# if __name__ == "__main__":
#     sample_chunks = [
#         "This document is extremely useful and well-written.",
#         "I hated the confusing structure of this report.",
#         "Overall, the results are acceptable and positive.",
#     ]

#     tagged = classify_chunks(sample_chunks)

#     print("\n--- Chunk Sentiment Results ---")
#     for item in tagged:
#         print(f"Chunk: {item['text']}")
#         print(f" → Sentiment: {item['sentiment']} (score={item['score']})\n")
