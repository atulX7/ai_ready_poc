import json
import pathlib
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set up paths and thresholds
PROCESSED = pathlib.Path("data/processed")
with open(PROCESSED / "metrics.json") as f:
    doc_scores = json.load(f)

HIGH_THRESHOLD = 0.76
LOW_THRESHOLD = 0.75

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

high, low = [], []

# For each PDF file in metrics.json
for score in doc_scores:
    base = score["file"].replace(".pdf", "")
    trust = score["ai_trust_score"]
    chunk_paths = sorted(PROCESSED.glob(f"{base}_*.txt"))

    for chunk_path in chunk_paths:
        text = chunk_path.read_text()
        emb = embedding_model.embed_query(text)

        if trust >= HIGH_THRESHOLD:
            high.append(emb)
        elif trust <= LOW_THRESHOLD:
            low.append(emb)

# Convert to numpy arrays
high_vecs = np.array(high)
low_vecs = np.array(low)

def avg_cosine_similarity(matrix):
    if len(matrix) < 2:
        return 0.0
    sim = cosine_similarity(matrix)
    tril = np.tril(sim, -1)  # lower triangle without diagonal
    return tril[tril != 0].mean()

print("\n🔍 Embedding Quality Validation (by trust score ranges):")
print(f"High-trust chunks: {len(high_vecs)}, Low-trust chunks: {len(low_vecs)}")

print(f"🔹 High Trust Avg Cosine Similarity: {avg_cosine_similarity(high_vecs):.3f}")
print(f"🔸 Low Trust Avg Cosine Similarity:  {avg_cosine_similarity(low_vecs):.3f}")
