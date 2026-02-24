from fastapi import FastAPI
from search_engine import HybridSearchEngine

app = FastAPI(title="Multilingual Hybrid Ad Template Search")

engine = HybridSearchEngine()

@app.get("/search")
def search(query: str, top_k: int = 5):
    results = engine.search(query, top_k)
    return {"results": results}