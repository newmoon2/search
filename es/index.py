import uuid
from typing import Any, Dict, Optional

from .common import INDEX_NAME, ensure_index, es, get_model


def index_text_to_es(texts: Dict[str, str], model_path: Optional[str] = None) -> Dict[str, Any]:
    """다중 텍스트 필드를 하나의 문서로 색인하고 각 필드별 임베딩 생성."""
    # 모델 가져오기
    model = get_model(model_path)
    
    ensure_index()

    # 입력 필드 정규화
    fields = {key: (texts.get(key) or "").strip() for key in ["text1", "text2", "text3", "text4", "text5"]}
    non_empty_values = [v for v in fields.values() if v]
    if not non_empty_values:
        raise ValueError("색인할 텍스트가 없습니다.")

    combined_text = "\n".join(non_empty_values)
    
    # 전체 텍스트에 대한 임베딩
    embedding = model.encode(combined_text).tolist()
    
    # 각 필드별 임베딩 생성
    embeddings = {}
    for i in range(1, 6):
        field_key = f"text{i}"
        field_value = fields.get(field_key, "")
        if field_value:
            embeddings[f"embedding{i}"] = model.encode(field_value).tolist()
    
    doc_id = str(uuid.uuid4())
    data = {**fields, "text": combined_text, "embedding": embedding, **embeddings}

    response = es.index(index=INDEX_NAME, id=doc_id, document=data)
    return {"doc_id": doc_id, "result": response}


if __name__ == "__main__":
    input_text = input("색인할 문자열 입력: ")
    res = index_text_to_es(input_text)
    print("색인 완료:", res)
