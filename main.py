from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

from es.index import index_text_to_es
from es.search import keyword_search, embedding_search, hybrid_search

app = FastAPI(title="Text Search API")

# 정적 디렉터리 생성 및 마운트 (추후 정적 파일 확장 시 사용)
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class IndexRequest(BaseModel):
    text1: Optional[str] = ""
    text2: Optional[str] = ""
    text3: Optional[str] = ""
    text4: Optional[str] = ""
    text5: Optional[str] = ""
    model_path: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    search_type: str = "hybrid"
    top_k: int = 5
    model_path: Optional[str] = None


@app.get("/", response_class=FileResponse)
async def read_root():
    """정적 HTML 파일 제공."""
    return FileResponse(static_dir / "index.html")


@app.get("/models")
async def list_models():
    """로컬 모델 목록 조회."""
    try:
        models = []
        
        # 기본 모델 추가
        models.append({
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "type": "huggingface"
        })
        
        # 일반적인 로컬 모델 경로들 확인
        common_paths = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path("models"),
            Path(".") / "models",
        ]
        
        for base_path in common_paths:
            if base_path.exists():
                for model_dir in base_path.iterdir():
                    if model_dir.is_dir():
                        # config.json이 있으면 모델로 간주
                        config_file = model_dir / "config.json"
                        if config_file.exists():
                            models.append({
                                "name": model_dir.name,
                                "path": str(model_dir.absolute()),
                                "type": "local"
                            })
        
        return {"models": models}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/index")
async def index_text(req: IndexRequest):
    try:
        texts = {k: v for k, v in req.model_dump().items() if k != "model_path"}
        return index_text_to_es(texts, req.model_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search")
async def search(req: SearchRequest):
    try:
        import json
        
        # 검색 타입에 따라 적절한 함수 호출
        if req.search_type == "keyword":
            hits, query_body = keyword_search(req.query, req.top_k, req.model_path)
        elif req.search_type == "embedding":
            hits, query_body = embedding_search(req.query, req.top_k, req.model_path)
        else:  # hybrid
            hits, query_body = hybrid_search(req.query, req.top_k, req.model_path)
        
        results = [
            {
                "id": hit.get("_id"),
                "score": hit.get("_score"),
                "text": hit["_source"].get("text"),
            }
            for hit in hits
        ]
        
        # 쿼리 body를 JSON 문자열로 변환
        query_json = json.dumps(query_body, indent=2, ensure_ascii=False)
        
        return {
            "count": len(results),
            "results": results,
            "query": query_json
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

