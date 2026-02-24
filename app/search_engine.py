import faiss
import numpy as np
import json
import pickle
import os
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi
try:
    from app.ranking import compute_recency_score, compute_usage_score, final_score
except ImportError:
    from ranking import compute_recency_score, compute_usage_score, final_score

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class HybridSearchEngine:

    def __init__(self):

        # Load embedding model once
        self.model = SentenceTransformer(MODEL_NAME)

        # Load FAISS index
        if not os.path.exists("storage/faiss.index"):
            raise ValueError("FAISS index not found. Run indexing.py first.")
        self.index = faiss.read_index("storage/faiss.index")

        # Load metadata
        with open("storage/metadata.json", "r") as f:
            self.metadata = json.load(f)

        # Load BM25
        with open("storage/bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

    def search(self, query: str, top_k: int = 5):

        if not query.strip():
            return []

        if len(self.metadata) == 0:
            return []

        top_k = min(top_k, len(self.metadata))

        # ---------------- VECTOR SEARCH ----------------
        query_vector = self.model.encode([query])
        query_vector = np.array(query_vector).astype("float32")
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, top_k)
        vector_scores = distances[0]

        # Normalize semantic scores (0 → 1)
        semantic_max = max(vector_scores) if len(vector_scores) > 0 else 1
        semantic_max = semantic_max if semantic_max != 0 else 1

        # ---------------- BM25 SEARCH ----------------
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_max = max(bm25_scores) if len(bm25_scores) > 0 else 1
        bm25_max = bm25_max if bm25_max != 0 else 1

        results = []

        for rank, idx in enumerate(indices[0]):

            template = self.metadata[idx]

            semantic_score = float(vector_scores[rank]) / semantic_max

            keyword_score = float(bm25_scores[idx]) / bm25_max

            recency_score = float(compute_recency_score(template["created_at"]))

            usage_score = float(compute_usage_score(template["usage_count"]))

            fuzzy_score = (
                fuzz.partial_ratio(
                    query.lower(),
                    template["title"].lower()
                ) / 100
            )

            keyword_combined = keyword_score + 0.1 * fuzzy_score

            total_score = final_score(
                semantic_score,
                keyword_combined,
                recency_score,
                usage_score
            )

            template_copy = template.copy()
            template_copy["score"] = round(total_score, 4)

            results.append(template_copy)

        # Final ranking
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results


    def add_template(self, new_template: dict):

        required_fields = ["id", "title", "description", "category", "created_at", "usage_count"]

        for field in required_fields:
            if field not in new_template:
                return {"error": f"Missing required field: {field}"}

        # Combine text
        combined_text = (
            new_template["title"] + " " +
            new_template["description"] + " " +
            new_template["category"]
        )

        new_embedding = self.model.encode([combined_text])
        new_embedding = np.array(new_embedding).astype("float32")
        faiss.normalize_L2(new_embedding)

        self.index.add(new_embedding)

        self.metadata.append(new_template)

        with open("storage/metadata.json", "w") as f:
            json.dump(self.metadata, f)

        
        faiss.write_index(self.index, "storage/faiss.index")

        
        tokenized_corpus = [
            (t["title"] + " " + t["description"] + " " + t["category"]).lower().split()
            for t in self.metadata
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        with open("storage/bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

        return {"status": "Template added successfully"}