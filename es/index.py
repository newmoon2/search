import uuid
from typing import Any, Dict

from .common import INDEX_NAME, ensure_index, es, model


def index_text_to_es(text: str) -> Dict[str, Any]:
    """텍스트를 임베딩과 함께 Elasticsearch에 색인."""
    ensure_index()

    embedding = model.encode(text).tolist()
    doc_id = str(uuid.uuid4())
    data = {"text": text, "embedding": embedding}

    response = es.index(index=INDEX_NAME, id=doc_id, document=data)
    return {"doc_id": doc_id, "result": response}


if __name__ == "__main__":
    input_text = input("색인할 문자열 입력: ")
    res = index_text_to_es(input_text)
    print("색인 완료:", res)
