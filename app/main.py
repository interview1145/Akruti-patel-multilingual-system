from fastapi import FastAPI
from pydantic import BaseModel
from search_engine import HybridSearchEngine

app = FastAPI(title="Multilingual Hybrid Ad Template Search")

engine = HybridSearchEngine()

@app.get("/search")
def search(query: str, top_k: int = 5):
    results = engine.search(query, top_k)
    return {"results": results}


class Template(BaseModel):
    id: str
    title: str
    description: str
    category: str
    created_at: str
    usage_count: int

@app.post("/add-template")
def add_template(template: Template):
    """
    Add a new ad template to the system.
    The template is indexed for semantic, keyword, and fuzzy search immediately.
    """
    result = engine.add_template(template.dict())
    return result