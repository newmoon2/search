import os
from pathlib import Path
from typing import Optional, Dict
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# 기본 설정값
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "R=uLP-jzCGGu+vdFDNQE")
INDEX_NAME = "index_nori_terms"
# LOCAL_MODEL_PATH = r"C:\0.project\dev\model\BGE-m3-ko"  # 로컬 임베딩 모델 경로
LOCAL_MODEL_PATH = r"C:\project\BGE-m3-ko"  # 로컬 임베딩 모델 경로
DEFAULT_MODEL = LOCAL_MODEL_PATH  # BGE-m3-ko를 기본 모델로 사용
# VECTOR_DIM은 기본 모델 로드 후 동적으로 설정됨
VECTOR_DIM = 1024  # BGE-m3-ko 임베딩 차원 (기본값)

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
try:
    default_model = SentenceTransformer(DEFAULT_MODEL)
    _model_cache[DEFAULT_MODEL] = default_model
    # 기본 모델의 차원으로 VECTOR_DIM 업데이트
    try:
        VECTOR_DIM = default_model.get_sentence_embedding_dimension()
    except:
        pass  # 기본값 유지
except Exception as e:
    # 모델 로드 실패 시 이전 기본 모델로 폴백
    print(f"Warning: Failed to load default model {DEFAULT_MODEL}: {e}")
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DIM = 384
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
    
    try:
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
    except Exception as e:
        raise ValueError(
            f"모델을 로드할 수 없습니다: {model_path}. 오류: {str(e)}"
        ) from e


def get_model_dimension(model_path: Optional[str] = None) -> int:
    """모델의 임베딩 차원을 반환."""
    model_instance = get_model(model_path)
    # 모델의 임베딩 차원 가져오기
    try:
        dim = model_instance.get_sentence_embedding_dimension()
        if dim is None or dim <= 0:
            raise ValueError(f"유효하지 않은 모델 차원: {dim}")
        return dim
    except Exception as e:
        # 차원을 가져올 수 없으면 오류 발생
        raise ValueError(
            f"모델의 임베딩 차원을 가져올 수 없습니다. model_path: {model_path}, 오류: {str(e)}"
        ) from e


def ensure_index(model_path: Optional[str] = None) -> None:
    """인덱스가 없으면 텍스트/임베딩 매핑과 함께 생성.
    모델 경로가 제공되면 해당 모델의 차원을 사용하여 인덱스를 생성/검증.
    """

    # 사용할 모델의 차원 확인
    model_dim = get_model_dimension(model_path)
    
    if model_dim is None or model_dim <= 0:
        raise ValueError(f"모델의 차원을 가져올 수 없습니다. model_path: {model_path}")
    
    # 인덱스가 이미 존재하는 경우 차원 확인
    if es.indices.exists(index=INDEX_NAME):
        try:
            # 기존 매핑 정보 가져오기
            mapping = es.indices.get_mapping(index=INDEX_NAME)[INDEX_NAME]["mappings"]
            existing_dims = mapping.get("properties", {}).get("embedding", {}).get("dims")

            # print(f"VECTOR_DIM: {VECTOR_DIM}")
            # print(f"model_dim: {model_dim}")
            # print(f"{INDEX_NAME} : 색인 완료..")
            
            # 차원이 일치하지 않으면 에러 발생
            if existing_dims is not None and existing_dims != model_dim:
                raise ValueError(
                    f"인덱스의 벡터 차원({existing_dims})과 모델의 차원({model_dim})이 일치하지 않습니다. "
                    f"인덱스를 삭제하고 다시 생성하거나, 동일한 차원의 모델을 사용하세요. "
                    f"인덱스 삭제 명령: es.indices.delete(index='{INDEX_NAME}')"
                )
        except ValueError:
            # 차원 불일치 오류는 재발생
            raise
        except Exception as e:
            # 매핑 정보를 가져오는 데 실패한 경우 - 상세한 오류 메시지와 함께 재발생
            raise ValueError(
                f"인덱스 매핑 정보를 확인하는 중 오류가 발생했습니다: {str(e)}. "
                f"인덱스를 삭제하고 다시 생성해보세요: es.indices.delete(index='{INDEX_NAME}')"
            ) from e
        return

    # 인덱스가 없으면 생성
    try:
        settings = {"index": {"knn": True}}
        mappings = {
            "dynamic_templates": [
                {
                    "embedding_fields": {
                        "match_mapping_type": "string",
                        "path_match": "*_embedding",
                        "mapping": {
                            "type": "dense_vector",
                            "dims": model_dim,
                            "index": "true",
                            "similarity": "cosine"
                        }
                    }
                }
            ],
            "properties": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": model_dim,
                    "index": True,
                    "similarity": "cosine",
                }
            }
        }

        es.indices.create(index=INDEX_NAME, settings=settings, mappings=mappings)
        print(f"인덱스 '{INDEX_NAME}'가 생성되었습니다 (벡터 차원: {model_dim}).")
    except Exception as e:
        error_msg = str(e)
        raise ValueError(
            f"인덱스 생성 중 오류가 발생했습니다: {error_msg}. "
            f"모델 차원: {model_dim}, 모델 경로: {model_path}"
        ) from e
