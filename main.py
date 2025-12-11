from pathlib import Path
import os
import csv
import io

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

from es.index import index_text_to_es
from es.search import keyword_search, embedding_search, hybrid_search
from es.common import INDEX_NAME, es

app = FastAPI(title="Search API")

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


class FileIndexRequest(BaseModel):
    """파일 경로를 통한 색인 요청."""
    file_path: str
    model_path: Optional[str] = None


class CsvRowData(BaseModel):
    """CSV 행 데이터 구조."""
    종목: str = ""
    세부종목: str = ""
    상품: str = ""
    증권코드: str = ""
    증권명: str = ""
    약관순번: str = ""
    약관코드: str = ""
    약관명: str = ""


class CsvIndexRequest(BaseModel):
    """CSV 행 데이터를 통한 색인 요청 (단일 행)."""
    data: CsvRowData
    model_path: Optional[str] = None


class CsvBatchIndexRequest(BaseModel):
    """CSV 행 데이터를 통한 색인 요청 (여러 행)."""
    data: List[CsvRowData]
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
        
        # 로컬 BGE-m3-ko 모델 추가
        from es.common import LOCAL_MODEL_PATH
        local_model_path = Path(LOCAL_MODEL_PATH)
        if local_model_path.exists():
            config_file = local_model_path / "config.json"
            if config_file.exists():
                models.append({
                    "name": "BGE-m3-ko",
                    "path": str(local_model_path.absolute()),
                    "type": "local"
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


@app.get("/index/info")
async def get_index_info():
    """인덱스 정보 조회 (매핑, 통계 등)."""
    try:
        if not es.indices.exists(index=INDEX_NAME):
            return {
                "exists": False,
                "message": f"인덱스 '{INDEX_NAME}'가 존재하지 않습니다."
            }
        
        # 매핑 정보 가져오기
        mapping = es.indices.get_mapping(index=INDEX_NAME)[INDEX_NAME]["mappings"]
        
        # 통계 정보 가져오기
        stats = es.indices.stats(index=INDEX_NAME)["indices"][INDEX_NAME]
        
        # 임베딩 차원 확인
        embedding_dims = {}
        properties = mapping.get("properties", {})
        for field_name, field_props in properties.items():
            if field_props.get("type") == "dense_vector":
                embedding_dims[field_name] = field_props.get("dims")
        
        return {
            "exists": True,
            "index_name": INDEX_NAME,
            "embedding_dimensions": embedding_dims,
            "document_count": stats["total"]["docs"]["count"],
            "index_size": stats["total"]["store"]["size_in_bytes"],
            "mapping": mapping
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/index")
async def delete_index():
    """인덱스 삭제."""
    try:
        if not es.indices.exists(index=INDEX_NAME):
            return {
                "message": f"인덱스 '{INDEX_NAME}'가 존재하지 않습니다.",
                "deleted": False
            }
        
        es.indices.delete(index=INDEX_NAME)
        return {
            "message": f"인덱스 '{INDEX_NAME}'가 삭제되었습니다.",
            "deleted": True
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def parse_csv_content(content: str) -> List[dict]:
    """CSV 내용을 파싱하여 딕셔너리 리스트로 반환."""
    csv_reader = csv.DictReader(io.StringIO(content))
    rows = []
    for row in csv_reader:
        rows.append(row)
    return rows


def map_csv_to_texts(csv_row: dict) -> dict:
    """
    CSV 행을 text1~text5 필드로 매핑.
    
    CSV 필드: 종목, 세부종목, 상품, 증권코드, 증권명, 약관순번, 약관코드, 약관명
    """
    # 빈 값 처리
    종목 = csv_row.get('종목', '').strip()
    세부종목 = csv_row.get('세부종목', '').strip()
    상품 = csv_row.get('상품', '').strip()
    증권코드 = csv_row.get('증권코드', '').strip()
    증권명 = csv_row.get('증권명', '').strip()
    약관순번 = csv_row.get('약관순번', '').strip()
    약관코드 = csv_row.get('약관코드', '').strip()
    약관명 = csv_row.get('약관명', '').strip()
    
    # text1~text5에 매핑
    texts = {
        "text1": f"{종목} {세부종목}".strip(),  # 종목 + 세부종목
        "text2": 상품,  # 상품
        "text3": f"{증권코드} {증권명}".strip(),  # 증권코드 + 증권명
        "text4": f"{약관순번} {약관코드}".strip(),  # 약관순번 + 약관코드
        "text5": 약관명  # 약관명
    }
    return texts


@app.post("/file_index")
async def index_file(
    file: UploadFile = File(...),
    model_path: Optional[str] = Form(None)
):
    """파일을 업로드하여 색인. CSV 파일인 경우 각 행을 개별 문서로 색인."""
    try:
        # 파일 내용 읽기
        content = await file.read()
        
        # 텍스트 디코딩 (UTF-8 시도, 실패 시 다른 인코딩 시도)
        try:
            text_content = content.decode('utf-8-sig')  # BOM 제거를 위해 utf-8-sig 사용
        except UnicodeDecodeError:
            try:
                text_content = content.decode('cp949')  # Windows 한글 인코딩
            except UnicodeDecodeError:
                text_content = content.decode('latin-1', errors='ignore')
        
        # 파일 확장자 확인
        filename = file.filename or ""
        is_csv = filename.lower().endswith('.csv')
        
        if is_csv:
            # CSV 파일 처리: 각 행을 개별 문서로 색인
            csv_rows = parse_csv_content(text_content)
            
            if not csv_rows:
                raise HTTPException(status_code=400, detail="CSV 파일에 데이터가 없습니다.")
            
            results = []
            for idx, row in enumerate(csv_rows, start=1):
                try:
                    texts = map_csv_to_texts(row)
                    result = index_text_to_es(texts, model_path)
                    results.append({
                        "row": idx,
                        "doc_id": result.get("doc_id"),
                        "status": "success"
                    })
                except Exception as e:
                    results.append({
                        "row": idx,
                        "status": "error",
                        "error": str(e)
                    })
            
            success_count = sum(1 for r in results if r.get("status") == "success")
            return {
                "message": f"CSV 파일 '{filename}' 색인 완료",
                "filename": filename,
                "file_size": len(content),
                "total_rows": len(csv_rows),
                "success_count": success_count,
                "error_count": len(csv_rows) - success_count,
                "results": results
            }
        else:
            # 일반 텍스트 파일 처리
            texts = {
                "text1": text_content,
                "text2": "",
                "text3": "",
                "text4": "",
                "text5": ""
            }
            
            result = index_text_to_es(texts, model_path)
            return {
                "message": f"파일 '{filename}' 색인 완료",
                "filename": filename,
                "file_size": len(content),
                **result
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/file_index/path")
async def index_file_by_path(req: FileIndexRequest):
    """파일 경로를 통해 파일을 읽어서 색인. CSV 파일인 경우 각 행을 개별 문서로 색인."""
    try:
        file_path = Path(req.file_path)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {req.file_path}")
        
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail=f"파일이 아닙니다: {req.file_path}")
        
        # 파일 읽기 (UTF-8 시도, 실패 시 다른 인코딩 시도)
        try:
            text_content = file_path.read_text(encoding='utf-8-sig')  # BOM 제거를 위해 utf-8-sig 사용
        except UnicodeDecodeError:
            try:
                text_content = file_path.read_text(encoding='cp949')  # Windows 한글 인코딩
            except UnicodeDecodeError:
                text_content = file_path.read_text(encoding='latin-1', errors='ignore')
        
        # 파일 확장자 확인
        is_csv = file_path.suffix.lower() == '.csv'
        
        if is_csv:
            # CSV 파일 처리: 각 행을 개별 문서로 색인
            csv_rows = parse_csv_content(text_content)
            
            if not csv_rows:
                raise HTTPException(status_code=400, detail="CSV 파일에 데이터가 없습니다.")
            
            results = []
            for idx, row in enumerate(csv_rows, start=1):
                try:
                    texts = map_csv_to_texts(row)
                    result = index_text_to_es(texts, req.model_path)
                    results.append({
                        "row": idx,
                        "doc_id": result.get("doc_id"),
                        "status": "success"
                    })
                except Exception as e:
                    results.append({
                        "row": idx,
                        "status": "error",
                        "error": str(e)
                    })
            
            success_count = sum(1 for r in results if r.get("status") == "success")
            return {
                "message": f"CSV 파일 '{req.file_path}' 색인 완료",
                "file_path": str(file_path.absolute()),
                "file_size": file_path.stat().st_size,
                "total_rows": len(csv_rows),
                "success_count": success_count,
                "error_count": len(csv_rows) - success_count,
                "results": results
            }
        else:
            # 일반 텍스트 파일 처리
            texts = {
                "text1": text_content,
                "text2": "",
                "text3": "",
                "text4": "",
                "text5": ""
            }
            
            result = index_text_to_es(texts, req.model_path)
            return {
                "message": f"파일 '{req.file_path}' 색인 완료",
                "file_path": str(file_path.absolute()),
                "file_size": file_path.stat().st_size,
                **result
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/csv_index/batch")
async def index_csv_rows_batch(req: CsvBatchIndexRequest):
    """CSV 행 데이터를 직접 받아서 색인 (여러 행 일괄 처리)."""
    try:
        if not req.data:
            raise HTTPException(status_code=400, detail="데이터가 없습니다.")
        
        results = []
        for idx, csv_row_data in enumerate(req.data, start=1):
            try:
                # CsvRowData를 딕셔너리로 변환
                csv_row = csv_row_data.model_dump()
                
                # text1~text5로 매핑
                texts = map_csv_to_texts(csv_row)
                
                # 색인 실행
                result = index_text_to_es(texts, req.model_path)
                results.append({
                    "index": idx,
                    "doc_id": result.get("doc_id"),
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "index": idx,
                    "status": "error",
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r.get("status") == "success")
        return {
            "message": f"CSV 행 데이터 일괄 색인 완료",
            "total_rows": len(req.data),
            "success_count": success_count,
            "error_count": len(req.data) - success_count,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/csv_index")
async def index_csv_row(req: CsvIndexRequest):
    """CSV 행 데이터를 직접 받아서 색인 (단일 행)."""
    try:
        # CsvRowData를 딕셔너리로 변환
        csv_row = req.data.model_dump()
        
        # text1~text5로 매핑
        texts = map_csv_to_texts(csv_row)
        
        # 색인 실행
        result = index_text_to_es(texts, req.model_path)
        
        return {
            "message": "CSV 행 데이터 색인 완료",
            **result
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

