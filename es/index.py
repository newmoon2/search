import uuid
from typing import Any, Dict, Optional

from .common import INDEX_NAME, ensure_index, es, get_model


def index_text_to_es(texts: Dict[str, str], model_path: Optional[str] = None) -> Dict[str, Any]:
    """다중 텍스트 필드를 하나의 문서로 색인하고 각 필드별 임베딩 생성."""
    from .common import get_model_dimension
    
    # 모델 가져오기
    model = get_model(model_path)
    
    # 모델 차원 확인
    model_dim = get_model_dimension(model_path)
    if model_dim is None or model_dim <= 0:
        raise ValueError(f"모델의 차원을 가져올 수 없습니다. model_path: {model_path}")
    
    # 인덱스 확인/생성 (모델 경로 전달하여 차원 검증)
    ensure_index(model_path)

    # 입력 필드 정규화
    fields = {key: (texts.get(key) or "").strip() for key in ["text1", "text2", "text3", "text4", "text5"]}
    non_empty_values = [v for v in fields.values() if v]
    if not non_empty_values:
        raise ValueError("색인할 텍스트가 없습니다.")

    combined_text = "\n".join(non_empty_values)
    
    # 전체 텍스트에 대한 임베딩
    embedding = model.encode(combined_text).tolist()
    
    # 임베딩 차원 검증
    if len(embedding) != model_dim:
        raise ValueError(
            f"임베딩 차원 불일치: 예상 차원={model_dim}, 실제 차원={len(embedding)}. "
            f"모델 경로: {model_path}"
        )
    
    # 각 필드별 임베딩 생성
    embeddings = {}
    for i in range(1, 6):
        field_key = f"text{i}"
        field_value = fields.get(field_key, "")
        if field_value:
            field_embedding = model.encode(field_value).tolist()
            # 각 필드 임베딩 차원도 검증
            if len(field_embedding) != model_dim:
                raise ValueError(
                    f"임베딩 차원 불일치 ({field_key}): 예상 차원={model_dim}, "
                    f"실제 차원={len(field_embedding)}. 모델 경로: {model_path}"
                )
            embeddings[f"embedding{i}"] = field_embedding
    
    doc_id = str(uuid.uuid4())
    data = {**fields, "text": combined_text, "embedding": embedding, **embeddings}

    try:
        response = es.index(index=INDEX_NAME, id=doc_id, document=data)
        return {"doc_id": doc_id, "result": response}
    except Exception as e:
        error_msg = str(e)
        if "mapper_parsing_exception" in error_msg.lower():
            # 상세한 오류 정보 포함
            raise ValueError(
                f"Elasticsearch 색인 오류 (mapper_parsing_exception): {error_msg}\n"
                f"모델 경로: {model_path}\n"
                f"모델 차원: {model_dim}\n"
                f"임베딩 차원: {len(embedding)}\n"
                f"인덱스: {INDEX_NAME}\n"
                f"인덱스를 삭제하고 다시 생성해보세요: es.indices.delete(index='{INDEX_NAME}')"
            ) from e
        raise ValueError(f"Elasticsearch 색인 오류: {error_msg}") from e


if __name__ == "__main__":
    input_text = input("색인할 문자열 입력: ")
    res = index_text_to_es(input_text)
    print("색인 완료:", res)
