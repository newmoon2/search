from pathlib import Path
import os
import csv
import io

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

from es.index import index_text_to_es, update_document_in_es
from es.search import keyword_search, embedding_search, hybrid_search, answer_keyword_search, answer_embedding_search, answer_hybrid_search
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
    search_text: str
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

class CsvRowDataAnswer(BaseModel):
    """정답지 CSV 행 데이터 구조."""
    uw_no: str = ""
    순번: str = ""
    약관명: str = ""
    약관코드: str = ""
    약관형태: str = ""
    약관형태코드: str = ""
    적용여부: str = ""
    증권명: str = ""   
    증권약관관계: str = ""
    증권코드: str = ""


class CsvRowDataUpdate(BaseModel):
    """상품 CSV 행 데이터 구조."""
    uw_no: str = ""
    종목: str = ""
    세부종목코드: str = ""
    세부종목명: str = ""
    상품코드: str = ""
    상품명: str = ""


class CsvBatchIndexRequest(BaseModel):
    """CSV 행 데이터를 통한 색인 요청 (여러 행)."""
    data: List[CsvRowData]
    model_path: Optional[str] = None

class CsvBatchIndexRequestAnswer(BaseModel):
    """정답지 CSV 행 데이터를 통한 색인 요청 (여러 행)."""
    data: List[CsvRowDataAnswer]
    model_path: Optional[str] = None    

class CsvBatchIndexRequestUpdate(BaseModel):
    """상품 CSV 행 데이터를 통한 색인 요청 (여러 행)."""
    data: List[CsvRowDataUpdate]
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
            hits, query_body = keyword_search(req.search_text, req.top_k, req.model_path)
        elif req.search_type == "embedding":
            hits, query_body = embedding_search(req.search_text, req.top_k, req.model_path)
        else:  # hybrid
            hits, query_body = hybrid_search(req.search_text, req.top_k, req.model_path)

        print(f"search_type : {req.search_type}")
        print(f"search_text : {req.search_text}")
        # print(f"search_type : {req.search_type}")
        
        results = [
            {
                "id": hit.get("_id"),
                "score": hit.get("_score"),
                "category_type": hit["_source"].get("category_type"),
                "sub_category": hit["_source"].get("sub_category"),
                "product": hit["_source"].get("product"),
                "security_code": hit["_source"].get("security_code"),
                "security_name": hit["_source"].get("security_name"),
                "clause_seq": hit["_source"].get("clause_seq"),
                "clause_code": hit["_source"].get("clause_code"),
                "clause_name": hit["_source"].get("clause_name"),
            }
            for hit in hits
        ]
        
        # 쿼리 body를 JSON 문자열로 변환
        query_json = json.dumps(query_body, indent=2, ensure_ascii=False)

        # print(f"count: len({results})")
        # print(f"results: {results}")
        
        return {
            "success": True,
            "count": len(results),
            "results": results,
            "query": query_json
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search/answer")
async def search(req: SearchRequest):
    """
    정답지 검색
    Docstring for search
    
    :param req: Description
    :type req: SearchRequest
    """
    try:
        import json
        
        # 검색 타입에 따라 적절한 함수 호출
        if req.search_type == "keyword":
            hits, query_body = answer_keyword_search(req.search_text, req.top_k, req.model_path)
        elif req.search_type == "embedding":
            hits, query_body = answer_embedding_search(req.search_text, req.top_k, req.model_path)
        else:  # hybrid
            hits, query_body = answer_hybrid_search(req.search_text, req.top_k, req.model_path)

        # print(f"search_type : {req.search_type}")
        # print(f"search_text : {req.search_text}")
        # print(f"search_type : {req.search_type}")
        
        results = [
            {
                "id": hit.get("_id"),
                "score": hit.get("_score"),
                "uw_no": hit["_source"].get("uw_no"),
                "order_no": hit["_source"].get("order_no"),
                "security_name": hit["_source"].get("security_name"),
                "security_code": hit["_source"].get("security_code"),
                "tc_name": hit["_source"].get("tc_name"),
                "tc_code": hit["_source"].get("tc_code"),
                "use_yn": hit["_source"].get("use_yn"),
                "tc_relation": hit["_source"].get("tc_relation"),
                "tc_form": hit["_source"].get("tc_form"),
                "tc_form_code": hit["_source"].get("tc_form_code"),
            }
            for hit in hits
        ]
        
        # 쿼리 body를 JSON 문자열로 변환
        query_json = json.dumps(query_body, indent=2, ensure_ascii=False)

        # print(f"count: len({results})")
        # print(f"results: {results}")
        
        return {
            "success": True,
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
    CSV 행을 text1~text5 및 도메인 필드로 매핑.
    
    CSV 필드: 종목, 세부종목, 상품, 증권코드, 증권명, 약관순번, 약관코드, 약관명
    -> 저장 필드: category_type, sub_category, product, security_code, security_name, clause_seq, clause_code, clause_name
    """
    # 빈 값 처리 및 필드명 매핑
    category_type = csv_row.get("종목", "").strip()
    sub_category = csv_row.get("세부종목", "").strip()
    product = csv_row.get("상품", "").strip()
    security_code = csv_row.get("증권코드", "").strip()
    security_name = csv_row.get("증권명", "").strip()
    clause_seq = csv_row.get("약관순번", "").strip()
    clause_code = csv_row.get("약관코드", "").strip()
    clause_name = csv_row.get("약관명", "").strip()
    
    # text1~text5에 매핑 + 도메인 필드 함께 저장
    texts = {
        # 검색 및 임베딩을 위한 조합/개별 필드
        "category_type": category_type,
        "sub_category": sub_category,
        "product": product,
        "security_code": security_code,
        "security_name": security_name,
        "clause_seq": clause_seq,
        "clause_code": clause_code,
        "clause_name": clause_name,
    }
    return texts


def map_answer_csv_to_texts(csv_row: dict) -> dict:
    """
    CSV 행을 text1~text5 및 도메인 필드로 매핑.
    
    CSV 필드: 'uw_no', '순번', '증권명', '증권코드', '약관명', '약관코드', '적용여부', '증권약관관계', '약관형태', '약관형태코드'
    -> 저장 필드: uw_no(uw_no),순번(order_no),증권명(security_name),증권코드(security_code),약관명(tc_name),약관코드(tc_code),
    적용여부(use_yn),증권약관관계(tc_relation),약관형태(tc_form),약관형태코드(tc_form_code)
    """
    # print(f"csv_row > {csv_row}")
    # 빈 값 처리 및 필드명 매핑
    uw_no = csv_row.get("uw_no", "").strip()
    order_no = csv_row.get("순번", "").strip()
    security_name = csv_row.get("증권명", "").strip()
    security_code = csv_row.get("증권코드", "").strip()
    tc_name = csv_row.get("약관명", "").strip()
    tc_code = csv_row.get("약관코드", "").strip()
    use_yn = csv_row.get("적용여부", "").strip()
    tc_relation = csv_row.get("증권약관관계", "").strip()
    tc_form = csv_row.get("약관형태", "").strip()
    tc_form_code = csv_row.get("약관형태코드", "").strip()
    
    # text1~text5에 매핑 + 도메인 필드 함께 저장
    texts = {
        # 검색 및 임베딩을 위한 조합/개별 필드
        "uw_no": uw_no,
        "order_no": order_no,
        "security_name": security_name,
        "security_code": security_code,
        "tc_name": tc_name,
        "tc_code": tc_code,
        "use_yn": use_yn,
        "tc_relation": tc_relation,
        "tc_form": tc_form,
        "tc_form_code": tc_form_code,
    }
    # print(f"texts : {texts}")
    return texts


def map_update_csv_to_texts(csv_row: dict) -> dict:
    """
    CSV 행을 text1~text5 및 도메인 필드로 매핑.
    
    CSV 필드: uw_no,종목,세부종목코드,세부종목명,상품코드,상품명
    -> 저장 필드: 종목(category_type),세부종목코드(sub_category),세부종목명(sub_category_name),상품코드(product_code),상품명(product_name)
    """
    print(f"csv_row > {csv_row}")
    # 빈 값 처리 및 필드명 매핑
    uw_no = csv_row.get("uw_no", "").strip()
    category_type = csv_row.get("종목", "").strip()
    sub_category = csv_row.get("세부종목코드", "").strip()
    sub_category_name = csv_row.get("세부종목명", "").strip()
    product_code = csv_row.get("상품코드", "").strip()
    product_name = csv_row.get("상품명", "").strip()

    # text1~text5에 매핑 + 도메인 필드 함께 저장
    texts = {
        # 검색 및 임베딩을 위한 조합/개별 필드
        "uw_no": uw_no,
        "category_type": category_type,
        "sub_category": sub_category,
        "sub_category_name": sub_category_name,
        "product_code": product_code,
        "product_name": product_name,
    }
    # print(f"texts : {texts}")
    return texts


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
                result = index_text_to_es(texts, req.model_path, "index_nori_terms")
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


@app.post("/csv_index/batch/answer")
async def index_csv_rows_batch_answer(req: CsvBatchIndexRequestAnswer):
    """정답지 CSV 행 데이터를 직접 받아서 색인 (여러 행 일괄 처리)."""
    try:
        if not req.data: 
            raise HTTPException(status_code=400, detail="데이터가 없습니다.")
        
        # print(f"req > {req}")

        results = []
        for idx, csv_row_data in enumerate(req.data, start=1):
            try:
                # CsvRowDataAnswer를 딕셔너리로 변환
                csv_row = csv_row_data.model_dump()
                
                # 정답지 데이터로 매핑
                # texts = map_csv_to_texts(csv_row)
                texts = map_answer_csv_to_texts(csv_row)
                
                # 색인실행
                result = index_text_to_es(texts, req.model_path, "index_nori_answer")

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
    

@app.post("/csv_index/batch/update")
async def index_csv_rows_batch_update(req: CsvBatchIndexRequestUpdate):
    """uw_no를 기준으로 CSV 행 데이터를 업데이트합니다."""
    try:
        if not req.data:
            raise HTTPException(status_code=400, detail="업데이트할 데이터가 없습니다.")

        results = []
        for idx, csv_row_data in enumerate(req.data, start=1):
            try:
                csv_row = csv_row_data.model_dump()
                
                if not csv_row.get("uw_no"):
                    raise ValueError("'uw_no'가 없는 행은 업데이트할 수 없습니다.")

                # CSV 데이터를 Elasticsearch 문서 형식으로 매핑
                texts = map_update_csv_to_texts(csv_row)
                
                # 업데이트 실행
                result = update_document_in_es(texts, req.model_path, "index_nori_answer")
                
                results.append({
                    "index": idx,
                    "uw_no": csv_row.get("uw_no"),
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                results.append({"index": idx, "uw_no": csv_row_data.uw_no, "status": "error", "error": str(e)})

        success_count = sum(1 for r in results if r["status"] == "success")
        return {
            "message": "CSV 데이터 일괄 업데이트 완료",
            "total_rows": len(req.data),
            "success_count": success_count,
            "error_count": len(req.data) - success_count,
            "results": results,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True, workers=4)
