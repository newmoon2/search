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
        "_source": ["uw_no","order_no","security_name","security_code","tc_name","tc_code","use_yn","tc_relation","tc_form","tc_form_code","category_type","sub_category","sub_category_name","product_code","product_name"],
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["uw_no","security_name","security_code","tc_name","tc_code","use_yn","tc_relation","tc_form","tc_form_code","category_type","sub_category","sub_category_name","product_code","product_name"],
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
                        "fields": ["uw_no","security_name","security_code","tc_name","tc_code","use_yn","tc_relation","tc_form","tc_form_code","category_type","sub_category","sub_category_name","product_code","product_name"],
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


def must_filter_conditions(query: str, top_k: int = 2, model_path: Optional[str] = None, category_type: str = "", sub_category: str = "", security_code: str = "", security_name: str = "", product_name: str = "", product_code: str = ""  ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """공통 must 및 filter 조건 생성 (미사용)."""
    must_conditions = []
    filter_conditions = []

    # ===== must (검색어) =====
    if query == "":
        must_conditions.append({
            "match_all": {}
        })
    else :
        must_conditions.append({
            "multi_match": {
                "query": query,
                "fields": ["category_type", "sub_category", "security_code", "security_name", "product_name", "product_code", "tc_code", "sub_category_name", "tc_form", "tc_form_code", "tc_name", "tc_relation", "uw_no"]
            }
        })
    # ===== filter (선택 조건들) =====
    if category_type:
        filter_conditions.append({
            "term": {
                "category_type.keyword": category_type
            }
        })

    if sub_category:
        filter_conditions.append({
            "term": {
                "sub_category.keyword": sub_category
            }
        })

    if security_code:
        filter_conditions.append({
            "term": {
                "security_code.keyword": security_code
            }
        })

    if security_name:
        filter_conditions.append({
            "match": {
                "security_name": security_name
            }
        })

    if product_code:
        filter_conditions.append({
            "term": {
                "product_code.keyword": product_code
            }
        })

    if product_name:
        filter_conditions.append({
            "match": {
                "product_name": product_name
            }
        })

    return must_conditions, filter_conditions



def test_keyword_search(query: str, top_k: int = 2, model_path: Optional[str] = None, category_type: str = "", sub_category: str = "", security_code: str = "", security_name: str = "", product_name: str = "", product_code: str = ""  ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """데이터 검증용 검색 """
    # ensure_index(model_path)

    # uw_no : 언더라이팅번호
    # category_type : 종목
    # sub_category : 세부종목
    # security_code : 증권코드
    # security_name : 증권명
    # product_name : 상품명
    # product_code : 상품코드

    print("test_keyword_search called")

    must_conditions, filter_conditions = must_filter_conditions(query, top_k, model_path, category_type, sub_category, security_code, security_name, product_name, product_code)

    body = {
        "size": top_k,
        "_source": [
            "uw_no", "order_no",
            "security_name", "security_code",
            "tc_name", "tc_code",
            "use_yn", "tc_relation",
            "tc_form", "tc_form_code",
            "category_type",
            "sub_category", "sub_category_name",
            "product_code", "product_name"
        ],
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                    "must": must_conditions,
                    "filter": filter_conditions
                    }
                },
                "script": {
                    "source": "( 2 / (1 + Math.exp(-1 * _score)) - 1 ) * 100"
                }
            }
        }
    }

    # print("ES Query Body:", json.dumps(body, indent=2, ensure_ascii=False))

    results = es.search(index="index_no_nori_test_2", body=body)
    return results["hits"]["hits"], body


def test_embedding_search(query: str, top_k: int = 2, model_path: Optional[str] = None, category_type: str = "", sub_category: str = "", security_code: str = "", security_name: str = "", product_name: str = "", product_code: str = ""  ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """코사인 유사도 임베딩 검색."""
    print("test_embedding_search called")
    # ensure_index(model_path)
    model = get_model(model_path)
    q_emb = model.encode(query).tolist()

    must_conditions, filter_conditions = must_filter_conditions(query, top_k, model_path, category_type, sub_category, security_code, security_name, product_name, product_code)

    # print("must_conditions:", must_conditions)    
    print("filter_conditions:", filter_conditions)

    body = {
        "size": top_k,

        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "must": {"match_all": {}},
                        "filter": filter_conditions
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
                        return maxSim + cosineSimilarity(params.query_vector, 'embedding');
                    """,
                    "params": {"query_vector": q_emb},
                },
            }
        }
    }

    results = es.search(index="index_no_nori_test_2", body=body)
    return results["hits"]["hits"], body


def test_hybrid_search(query: str, top_k: int = 2, model_path: Optional[str] = None, category_type: str = "", sub_category: str = "", security_code: str = "", security_name: str = "", product_name: str = "", product_code: str = ""  ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """BM25 + 코사인 유사도 하이브리드 검색"""
    print("test_hybrid_search called")
    # ensure_index(model_path)
    model = get_model(model_path)
    q_emb = model.encode(query).tolist()

    must_conditions, filter_conditions = must_filter_conditions(query, top_k, model_path, category_type, sub_category, security_code, security_name, product_name, product_code)

    # 검색 비율
    keyword_weight = 0.5
    vector_weight = 0.5

    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "must": must_conditions,
                        "filter": filter_conditions,
                    },
                },
                "script": {
                    "source": """
                    double maxSim = 0.0;

                    // 여러 embedding 필드 중 최대 cosine similarity
                    for (key in params._source.keySet()) {
                    if (key.endsWith('_embedding') && doc[key].size() > 0) {
                        maxSim = Math.max(
                        maxSim,
                        cosineSimilarity(params.query_vector, key)
                        );
                    }
                    }

                    
                    double keywordScore = Math.log(1 + _score);

                    
                    double vectorScore = (maxSim + 1.0) / 2.0;

                    
                    double finalScore =
                        params.keyword_weight * keywordScore +
                        params.vector_weight  * vectorScore;

                    
                    finalScore = Math.min(1.0, Math.max(0.0, finalScore));

                    
                    return finalScore * 100;
                """,
                    "params": {
                        "query_vector": q_emb,
                        "keyword_weight": keyword_weight,
                        "vector_weight": vector_weight
                    }
                # "script": {
                #     "source": """
                #     double maxSim = 0.0;
                #     for (key in params._source.keySet()) {
                #         if (key.endsWith('_embedding') && doc[key].size() > 0) {
                #             maxSim = Math.max(
                #                 maxSim,
                #                 cosineSimilarity(params.query_vector, key)
                #             );
                #         }
                #     }

                #     double vectorScore = maxSim;
                #     double keywordScore = _score;

                #     return
                #         (params.keyword_weight * keywordScore)
                #       + (params.vector_weight * vectorScore);
                #     """,
                #     "params": {
                #         "query_vector": q_emb,
                #         "keyword_weight": keyword_weight,
                #         "vector_weight": vector_weight
                #     }
                # },
                }
            }
        },
    }

    results = es.search(index="index_no_nori_test_2", body=body, timeout="60s")
    return results["hits"]["hits"], body


if __name__ == "__main__":
    result, query_body = hybrid_search(input("검색어 입력: "))
    for r in result:
        print(r["_source"]["text"], r["_score"])
