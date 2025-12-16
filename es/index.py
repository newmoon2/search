import uuid
from typing import Any, Dict, Optional

from .common import INDEX_NAME, ensure_index, es, get_model


def index_text_to_es(texts: Dict[str, str], model_path: Optional[str] = None, index: str = "index_nori_terms") -> Dict[str, Any]:
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

    # 텍스트 필드만 추출 (값이 있는 경우)
    text_fields = {k: v.strip() for k, v in texts.items() if isinstance(v, str) and v.strip()}

    non_empty_values = list(text_fields.values())
    if not non_empty_values:
        raise ValueError("색인할 텍스트가 없습니다.")

    combined_text = "\n".join(non_empty_values)

    # 색인할 문서 준비
    doc_id = str(uuid.uuid4())
    data = {
        **texts,  # 원본 필드들 (product, clause_name 등)
        "text": combined_text, # 모든 텍스트 필드를 합친 필드
    }

    # 임베딩 생성 (전체 텍스트 + 각 개별 필드)
    texts_to_embed = {"": combined_text, **text_fields}
    embeddings_list = model.encode(list(texts_to_embed.values())).tolist()

    # 생성된 임베딩을 문서에 추가
    for i, (key, _) in enumerate(texts_to_embed.items()):
        embedding = embeddings_list[i]
        if len(embedding) != model_dim:
            field_name = f"'{key}'" if key else "combined text"
            raise ValueError(
                f"임베딩 차원 불일치 ({field_name}): 예상 차원={model_dim}, 실제 차원={len(embedding)}"
            )
        
        embedding_key = f"{key}_embedding" if key else "embedding"
        data[embedding_key] = embedding

    try:
        response = es.index(index=index, id=doc_id, document=data)
        print("색인 완료 > ", doc_id)
        return {"doc_id": doc_id, "result": response}
    except Exception as e:
        error_msg = str(e)
        raise ValueError(f"Elasticsearch 색인 오류: {error_msg}") from e


if __name__ == "__main__":
    input_text = input("색인할 문자열 입력: ")
    res = index_text_to_es(input_text)
    print("색인 완료:", res)
