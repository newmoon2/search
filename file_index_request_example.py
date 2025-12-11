"""
외부에서 /file_index 엔드포인트를 호출하는 예제 코드

CSV 파일 형식:
--------------
CSV 파일은 다음 8개 필드를 포함해야 합니다:
종목,세부종목,상품,증권코드,증권명,약관순번,약관코드,약관명

예제 CSV:
종목,세부종목,상품,증권코드,증권명,약관순번,약관코드,약관명
주식,일반,상장주식,005930,삼성전자,1,TERM001,기본약관

CURL 예제:
----------
1. CSV 파일 업로드 방식:
   curl -X POST "http://localhost:8000/file_index" \
     -F "file=@/path/to/your/file.csv" \
     -F "model_path=C:/0.project/dev/model/BGE-m3-ko"

2. CSV 파일 경로 방식:
   curl -X POST "http://localhost:8000/file_index/path" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "C:/path/to/your/file.csv", "model_path": "C:/0.project/dev/model/BGE-m3-ko"}'
"""

import requests
from pathlib import Path
from typing import Optional


# API 기본 URL
BASE_URL = "http://localhost:8000"


def index_file_upload(file_path: str, model_path: Optional[str] = None) -> dict:
    """
    파일을 업로드하여 색인하는 함수
    CSV 파일인 경우 각 행을 개별 문서로 색인합니다.
    
    Args:
        file_path: 업로드할 파일의 경로 (CSV 또는 텍스트 파일)
        model_path: 사용할 모델 경로 (선택사항)
    
    Returns:
        API 응답 결과
        CSV 파일인 경우: total_rows, success_count, error_count, results 포함
        일반 파일인 경우: doc_id, result 포함
    """
    url = f"{BASE_URL}/file_index"
    
    # 파일 MIME 타입 결정
    file_ext = Path(file_path).suffix.lower()
    content_type = 'text/csv' if file_ext == '.csv' else 'text/plain'
    
    # 파일 열기
    with open(file_path, 'rb') as f:
        files = {
            'file': (Path(file_path).name, f, content_type)
        }
        
        data = {}
        if model_path:
            data['model_path'] = model_path
        
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()


def index_file_by_path(file_path: str, model_path: Optional[str] = None) -> dict:
    """
    파일 경로를 전달하여 색인하는 함수
    CSV 파일인 경우 각 행을 개별 문서로 색인합니다.
    
    Args:
        file_path: 색인할 파일의 경로 (서버에서 접근 가능한 경로, CSV 또는 텍스트 파일)
        model_path: 사용할 모델 경로 (선택사항)
    
    Returns:
        API 응답 결과
        CSV 파일인 경우: total_rows, success_count, error_count, results 포함
        일반 파일인 경우: doc_id, result 포함
    """
    url = f"{BASE_URL}/file_index/path"
    
    payload = {
        "file_path": file_path,
    }
    if model_path:
        payload["model_path"] = model_path
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


# 사용 예제
if __name__ == "__main__":
    # 예제 1: CSV 파일 업로드 방식
    print("=== CSV 파일 업로드 방식 ===")
    try:
        result = index_file_upload("example.csv")
        print(f"성공: {result}")
        if "total_rows" in result:
            print(f"  - 총 행 수: {result['total_rows']}")
            print(f"  - 성공: {result['success_count']}, 실패: {result['error_count']}")
    except Exception as e:
        print(f"오류: {e}")
    
    # 예제 2: CSV 파일 경로 방식
    print("\n=== CSV 파일 경로 방식 ===")
    try:
        result = index_file_by_path("C:/path/to/your/file.csv")
        print(f"성공: {result}")
        if "total_rows" in result:
            print(f"  - 총 행 수: {result['total_rows']}")
            print(f"  - 성공: {result['success_count']}, 실패: {result['error_count']}")
    except Exception as e:
        print(f"오류: {e}")
    
    # 예제 3: 모델 경로 지정 (CSV)
    print("\n=== 모델 경로 지정 (CSV) ===")
    try:
        result = index_file_upload(
            "example.csv",
            model_path="C:/0.project/dev/model/BGE-m3-ko"
        )
        print(f"성공: {result}")
        if "total_rows" in result:
            print(f"  - 총 행 수: {result['total_rows']}")
            print(f"  - 성공: {result['success_count']}, 실패: {result['error_count']}")
    except Exception as e:
        print(f"오류: {e}")
    
    # 예제 4: 일반 텍스트 파일
    print("\n=== 일반 텍스트 파일 ===")
    try:
        result = index_file_upload("example.txt")
        print(f"성공: {result}")
    except Exception as e:
        print(f"오류: {e}")

