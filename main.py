from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from es.index import index_text_to_es
from es.search import hybrid_search

app = FastAPI(title="Text Search API")


class IndexRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/index")
async def index_text(req: IndexRequest):
    try:
        return index_text_to_es(req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search")
async def search(req: SearchRequest):
    try:
        hits = hybrid_search(req.query, req.top_k)
        results = [
            {
                "id": hit.get("_id"),
                "score": hit.get("_score"),
                "text": hit["_source"].get("text"),
            }
            for hit in hits
        ]
        return {"count": len(results), "results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

