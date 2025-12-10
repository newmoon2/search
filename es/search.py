from typing import Any, Dict, List

from .common import INDEX_NAME, ensure_index, es, model


def hybrid_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """BM25 + 코사인 유사도 하이브리드 검색."""
    ensure_index()
    q_emb = model.encode(query).tolist()

    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text"],
                    }
                },
                "script": {
                    "source": "(_score) + (2 * cosineSimilarity(params.query_vector, 'embedding'))",
                    "params": {"query_vector": q_emb},
                },
            }
        },
    }

    results = es.search(index=INDEX_NAME, body=body)
    return results["hits"]["hits"]


if __name__ == "__main__":
    result = hybrid_search(input("검색어 입력: "))
    for r in result:
        print(r["_source"]["text"], r["_score"])
