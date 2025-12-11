from typing import Any, Dict, List, Tuple, Optional
import json

from .common import INDEX_NAME, ensure_index, es, get_model


def keyword_search(query: str, top_k: int = 5, model_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """BM25 키워드 검색."""
    ensure_index(model_path)

    body = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["text", "text1", "text2", "text3", "text4", "text5"],
            }
        },
    }

    results = es.search(index=INDEX_NAME, body=body)
    return results["hits"]["hits"], body


def embedding_search(query: str, top_k: int = 5, model_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """코사인 유사도 임베딩 검색."""
    ensure_index(model_path)
    model = get_model(model_path)
    q_emb = model.encode(query).tolist()

    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": """
                        double maxSim = cosineSimilarity(params.query_vector, 'embedding');
                        if (doc['embedding1'].size() > 0) {
                            maxSim = Math.max(maxSim, cosineSimilarity(params.query_vector, 'embedding1'));
                        }
                        if (doc['embedding2'].size() > 0) {
                            maxSim = Math.max(maxSim, cosineSimilarity(params.query_vector, 'embedding2'));
                        }
                        if (doc['embedding3'].size() > 0) {
                            maxSim = Math.max(maxSim, cosineSimilarity(params.query_vector, 'embedding3'));
                        }
                        if (doc['embedding4'].size() > 0) {
                            maxSim = Math.max(maxSim, cosineSimilarity(params.query_vector, 'embedding4'));
                        }
                        if (doc['embedding5'].size() > 0) {
                            maxSim = Math.max(maxSim, cosineSimilarity(params.query_vector, 'embedding5'));
                        }
                        return maxSim;
                    """,
                    "params": {"query_vector": q_emb},
                },
            }
        },
    }

    results = es.search(index=INDEX_NAME, body=body)
    return results["hits"]["hits"], body


def hybrid_search(query: str, top_k: int = 5, model_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """BM25 + 코사인 유사도 하이브리드 검색 (각 필드별 임베딩 고려)."""
    ensure_index(model_path)
    model = get_model(model_path)
    q_emb = model.encode(query).tolist()

    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text", "text1", "text2", "text3", "text4", "text5"],
                    }
                },
                "script": {
                    "source": """
                        double maxSim = cosineSimilarity(params.query_vector, 'embedding');
                        if (doc['embedding1'].size() > 0) {
                            maxSim = Math.max(maxSim, cosineSimilarity(params.query_vector, 'embedding1'));
                        }
                        if (doc['embedding2'].size() > 0) {
                            maxSim = Math.max(maxSim, cosineSimilarity(params.query_vector, 'embedding2'));
                        }
                        if (doc['embedding3'].size() > 0) {
                            maxSim = Math.max(maxSim, cosineSimilarity(params.query_vector, 'embedding3'));
                        }
                        if (doc['embedding4'].size() > 0) {
                            maxSim = Math.max(maxSim, cosineSimilarity(params.query_vector, 'embedding4'));
                        }
                        if (doc['embedding5'].size() > 0) {
                            maxSim = Math.max(maxSim, cosineSimilarity(params.query_vector, 'embedding5'));
                        }
                        return _score + (2 * maxSim);
                    """,
                    "params": {"query_vector": q_emb},
                },
            }
        },
    }

    results = es.search(index=INDEX_NAME, body=body)
    return results["hits"]["hits"], body


if __name__ == "__main__":
    result, query_body = hybrid_search(input("검색어 입력: "))
    for r in result:
        print(r["_source"]["text"], r["_score"])
