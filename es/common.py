import os
from pathlib import Path
from typing import Optional, Dict
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# 기본 설정값
ES_HOST = os.getenv("ES_HOST", "https://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "R=uLP-jzCGGu+vdFDNQE")
INDEX_NAME = "index_nori"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 임베딩 차원
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# SSL 설정: 자체 서명 인증서 사용 시 환경변수로 제어
# ES_VERIFY_CERTS = os.getenv("ES_VERIFY_CERTS", "true").lower() == "false"
# ES_CA_CERT = os.getenv("ES_CA_CERT")  # CA 번들 경로 (선택)

# 전역 클라이언트/모델 초기화
es = Elasticsearch(
    ES_HOST,
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False,
    ca_certs=None,
)

# 모델 캐시 딕셔너리
_model_cache: Dict[str, SentenceTransformer] = {}

# 기본 모델 로드
default_model = SentenceTransformer(DEFAULT_MODEL)
_model_cache[DEFAULT_MODEL] = default_model
model = default_model


def get_model(model_path: Optional[str] = None) -> SentenceTransformer:
    """모델 경로를 받아서 모델을 반환. 캐시에 있으면 재사용, 없으면 로드."""
    if model_path is None or model_path == "":
        return default_model
    
    # 캐시에 있으면 반환
    if model_path in _model_cache:
        return _model_cache[model_path]
    
    # 로컬 경로인지 확인
    if os.path.exists(model_path) or Path(model_path).exists():
        # 로컬 경로에서 로드
        loaded_model = SentenceTransformer(model_path)
    else:
        # HuggingFace 모델 이름으로 로드
        loaded_model = SentenceTransformer(model_path)
    
    # 캐시에 저장
    _model_cache[model_path] = loaded_model
    return loaded_model


def get_model_dimension(model_path: Optional[str] = None) -> int:
    """모델의 임베딩 차원을 반환."""
    model_instance = get_model(model_path)
    # 모델의 임베딩 차원 가져오기
    try:
        return model_instance.get_sentence_embedding_dimension()
    except:
        # 기본값 반환
        return VECTOR_DIM


def ensure_index() -> None:
    """인덱스가 없으면 텍스트/임베딩 매핑과 함께 생성."""
    if es.indices.exists(index=INDEX_NAME):
        return

    settings = {"index": {"knn": True}}
    mappings = {
        "properties": {
            "text": {"type": "text"},
            "text1": {"type": "text"},
            "text2": {"type": "text"},
            "text3": {"type": "text"},
            "text4": {"type": "text"},
            "text5": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": VECTOR_DIM,
                "index": True,
                "similarity": "cosine",
            },
            "embedding1": {
                "type": "dense_vector",
                "dims": VECTOR_DIM,
                "index": True,
                "similarity": "cosine",
            },
            "embedding2": {
                "type": "dense_vector",
                "dims": VECTOR_DIM,
                "index": True,
                "similarity": "cosine",
            },
            "embedding3": {
                "type": "dense_vector",
                "dims": VECTOR_DIM,
                "index": True,
                "similarity": "cosine",
            },
            "embedding4": {
                "type": "dense_vector",
                "dims": VECTOR_DIM,
                "index": True,
                "similarity": "cosine",
            },
            "embedding5": {
                "type": "dense_vector",
                "dims": VECTOR_DIM,
                "index": True,
                "similarity": "cosine",
            },
        }
    }

    es.indices.create(index=INDEX_NAME, settings=settings, mappings=mappings)

