import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
import os

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def build_indexes():

    model = SentenceTransformer(MODEL_NAME)

    with open("data/templates.json", "r") as f:
        templates = json.load(f)

    texts = [
        t["title"] + " " + t["description"] + " " + t["category"]
        for t in templates
    ]

    # -------- VECTOR EMBEDDINGS --------
    embeddings = model.encode(texts, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    os.makedirs("storage", exist_ok=True)

    faiss.write_index(index, "storage/faiss.index")

    # -------- BM25 --------
    tokenized_corpus = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    with open("storage/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # Save metadata
    with open("storage/metadata.json", "w") as f:
        json.dump(templates, f)

    print("Indexes built successfully!")


if __name__ == "__main__":
    build_indexes()