from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# 기본 설정값
ES_HOST = "http://localhost:9200"
ES_USER = "elastic"
ES_PASSWORD = "your_password"
INDEX_NAME = "text_index"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 임베딩 차원

# 전역 클라이언트/모델 초기화
es = Elasticsearch(ES_HOST, basic_auth=(ES_USER, ES_PASSWORD))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def ensure_index() -> None:
    """인덱스가 없으면 텍스트/임베딩 매핑과 함께 생성."""
    if es.indices.exists(index=INDEX_NAME):
        return

    settings = {"index": {"knn": True}}
    mappings = {
        "properties": {
            "text": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": VECTOR_DIM,
                "index": True,
                "similarity": "cosine",
            },
        }
    }

    es.indices.create(index=INDEX_NAME, settings=settings, mappings=mappings)

