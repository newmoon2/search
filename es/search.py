from typing import Any, Dict, List, Tuple, Optional
import json

from .common import INDEX_NAME, ensure_index, es, get_model


def keyword_search(query: str, top_k: int = 2, model_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """BM25 키워드 검색."""
    ensure_index(model_path)

    body = {
        "size": top_k,
        "_source": ["category_type", "sub_category", "product", "security_code", "security_name", "clause_seq","clause_code","clause_name"],
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["category_type", "sub_category", "product", "security_code", "security_name", "clause_seq","clause_code","clause_name"],
                # "fields": ["clause_name"],
            }
        },
    }

    results = es.search(index=INDEX_NAME, body=body)
    return results["hits"]["hits"], body


def embedding_search(query: str, top_k: int = 2, model_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
                        double maxSim = 0.0;
                        for (key in params._source.keySet()) {
                            if (key.endsWith('_embedding')) {
                                if (doc[key].size() > 0) {
                                    maxSim = Math.max(maxSim, cosineSimilarity(params.query_vector, key));
                                }
                            }
                        }
                        return maxSim + cosineSimilarity(params.query_vector, 'embedding');
                    """,
                    "params": {"query_vector": q_emb},
                },
            }
        },
    }

    results = es.search(index=INDEX_NAME, body=body)
    return results["hits"]["hits"], body


def hybrid_search(query: str, top_k: int = 2, model_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
                        "fields": ["category_type", "sub_category", "product", "security_code", "security_name", "clause_seq","clause_code","clause_name"],
                    }
                },
                "script": {
                    "source": """
                        double maxSim = 0.0;
                        for (key in params._source.keySet()) {
                            if (key.endsWith('_embedding')) {
                                if (doc[key].size() > 0) {
                                    maxSim = Math.max(maxSim, cosineSimilarity(params.query_vector, key));
                                }
                            }
                        }
                        return _score + (2 * (maxSim + cosineSimilarity(params.query_vector, 'embedding')));
                    """,
                    "params": {"query_vector": q_emb},
                },
            }
        },
    }

    results = es.search(index=INDEX_NAME, body=body)
    return results["hits"]["hits"], body


def answer_keyword_search(query: str, top_k: int = 2, model_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """BM25 키워드 검색."""
    ensure_index(model_path)

    body = {
        "size": top_k,
        "_source": ["uw_no","order_no","security_name","security_code","tc_name","tc_code","use_yn","tc_relation","tc_form","tc_form_code"],
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["uw_no","security_name","security_code","tc_name","tc_code","use_yn","tc_relation","tc_form","tc_form_code"],
            }
        },
    }

    results = es.search(index="index_nori_answer", body=body)
    return results["hits"]["hits"], body


def answer_embedding_search(query: str, top_k: int = 2, model_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
                        double maxSim = 0.0;
                        for (key in params._source.keySet()) {
                            if (key.endsWith('_embedding')) {
                                if (doc[key].size() > 0) {
                                    maxSim = Math.max(maxSim, cosineSimilarity(params.query_vector, key));
                                }
                            }
                        }
                        return maxSim + cosineSimilarity(params.query_vector, 'embedding');
                    """,
                    "params": {"query_vector": q_emb},
                },
            }
        },
    }

    results = es.search(index="index_nori_answer", body=body)
    return results["hits"]["hits"], body


def answer_hybrid_search(query: str, top_k: int = 2, model_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
                        "fields": ["uw_no","security_name","security_code","tc_name","tc_code","use_yn","tc_relation","tc_form","tc_form_code"],
                    }
                },
                "script": {
                    "source": """
                        double maxSim = 0.0;
                        for (key in params._source.keySet()) {
                            if (key.endsWith('_embedding')) {
                                if (doc[key].size() > 0) {
                                    maxSim = Math.max(maxSim, cosineSimilarity(params.query_vector, key));
                                }
                            }
                        }
                        return _score + (2 * (maxSim + cosineSimilarity(params.query_vector, 'embedding')));
                    """,
                    "params": {"query_vector": q_emb},
                },
            }
        },
    }

    results = es.search(index="index_nori_answer", body=body)
    return results["hits"]["hits"], body

if __name__ == "__main__":
    result, query_body = hybrid_search(input("검색어 입력: "))
    for r in result:
        print(r["_source"]["text"], r["_score"])
