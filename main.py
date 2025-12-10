from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from es.index import index_text_to_es
from es.search import hybrid_search

app = FastAPI(title="Text Search API")

# 정적 디렉터리 생성 및 마운트 (추후 정적 파일 확장 시 사용)
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class IndexRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """간단한 웹 UI 제공."""
    html_content = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>텍스트 검색 시스템</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }
        .container {
            max-width: 1200px; margin: 0 auto; background: #fff;
            border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); padding: 40px;
        }
        h1 { color: #333; text-align: center; margin-bottom: 40px; font-size: 2.5em; }
        .section { margin-bottom: 40px; padding: 30px; background: #f8f9fa; border-radius: 15px; border: 2px solid #e9ecef; }
        .section h2 { color: #667eea; margin-bottom: 20px; font-size: 1.8em; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #555; font-weight: 600; }
        textarea, input[type="text"], input[type="number"] {
            width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; transition: border-color 0.3s;
        }
        textarea { min-height: 120px; resize: vertical; }
        textarea:focus, input:focus { outline: none; border-color: #667eea; }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #fff;
            border: none; padding: 14px 30px; border-radius: 8px; font-size: 16px; font-weight: 600;
            cursor: pointer; transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
        button:active { transform: translateY(0); }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .message { margin-top: 15px; padding: 12px; border-radius: 8px; display: none; }
        .message.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .message.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .results { margin-top: 20px; }
        .result-item {
            background: #fff; padding: 20px; margin-bottom: 15px; border-radius: 10px;
            border-left: 4px solid #667eea; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .result-item .score { color: #667eea; font-weight: 600; margin-bottom: 10px; }
        .result-item .text { color: #333; line-height: 1.6; }
        .result-count { color: #666; margin-bottom: 15px; font-size: 1.1em; }
        .loading { display: none; text-align: center; color: #667eea; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 텍스트 검색 시스템</h1>

        <div class="section">
            <h2>📝 텍스트 색인</h2>
            <div class="form-group">
                <label for="indexText">색인할 텍스트 입력:</label>
                <textarea id="indexText" placeholder="색인하고 싶은 텍스트를 입력하세요..."></textarea>
            </div>
            <button onclick="indexText()">색인하기</button>
            <div id="indexMessage" class="message"></div>
            <div id="indexLoading" class="loading">색인 중...</div>
        </div>

        <div class="section">
            <h2>🔎 텍스트 검색</h2>
            <div class="form-group">
                <label for="searchQuery">검색어 입력:</label>
                <input type="text" id="searchQuery" placeholder="검색하고 싶은 키워드를 입력하세요...">
            </div>
            <div class="form-group">
                <label for="topK">결과 개수:</label>
                <input type="number" id="topK" value="5" min="1" max="20">
            </div>
            <button onclick="searchText()">검색하기</button>
            <div id="searchMessage" class="message"></div>
            <div id="searchLoading" class="loading">검색 중...</div>
            <div id="searchResults" class="results"></div>
        </div>
    </div>

    <script>
        async function indexText() {
            const text = document.getElementById('indexText').value.trim();
            const loading = document.getElementById('indexLoading');
            const msg = document.getElementById('indexMessage');
            if (!text) { showMessage('indexMessage', '텍스트를 입력해주세요.', 'error'); return; }
            loading.style.display = 'block'; msg.style.display = 'none';
            try {
                const res = await fetch('/index', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const data = await res.json();
                if (res.ok) {
                    showMessage('indexMessage', '색인이 완료되었습니다! (ID: ' + data.doc_id + ')', 'success');
                    document.getElementById('indexText').value = '';
                } else {
                    showMessage('indexMessage', '색인 실패: ' + (data.detail || '알 수 없는 오류'), 'error');
                }
            } catch (e) {
                showMessage('indexMessage', '오류 발생: ' + e.message, 'error');
            } finally { loading.style.display = 'none'; }
        }

        async function searchText() {
            const query = document.getElementById('searchQuery').value.trim();
            const topK = parseInt(document.getElementById('topK').value) || 5;
            const loading = document.getElementById('searchLoading');
            const msg = document.getElementById('searchMessage');
            const resultsDiv = document.getElementById('searchResults');
            if (!query) { showMessage('searchMessage', '검색어를 입력해주세요.', 'error'); return; }
            loading.style.display = 'block'; msg.style.display = 'none'; resultsDiv.innerHTML = '';
            try {
                const res = await fetch('/search', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, top_k: topK })
                });
                const data = await res.json();
                if (res.ok) {
                    if (data.count === 0) showMessage('searchMessage', '검색 결과가 없습니다.', 'error');
                    else { showMessage('searchMessage', data.count + '개의 결과를 찾았습니다.', 'success'); displayResults(data.results); }
                } else {
                    showMessage('searchMessage', '검색 실패: ' + (data.detail || '알 수 없는 오류'), 'error');
                }
            } catch (e) {
                showMessage('searchMessage', '오류 발생: ' + e.message, 'error');
            } finally { loading.style.display = 'none'; }
        }

        function displayResults(results) {
            const div = document.getElementById('searchResults');
            let html = '<div class="result-count">총 ' + results.length + '개의 결과</div>';
            results.forEach(r => {
                html += `
                    <div class="result-item">
                        <div class="score">유사도 점수: ${r.score.toFixed(4)}</div>
                        <div class="text">${escapeHtml(r.text)}</div>
                    </div>
                `;
            });
            div.innerHTML = html;
        }

        function showMessage(id, message, type) {
            const el = document.getElementById(id);
            el.textContent = message;
            el.className = 'message ' + type;
            el.style.display = 'block';
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        document.getElementById('searchQuery').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') searchText();
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post("/index")
async def index_text(req: IndexRequest):
    try:
        return index_text_to_es(req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search")
async def search(req: SearchRequest):
    try:
        hits = hybrid_search(req.query, req.top_k)
        results = [
            {
                "id": hit.get("_id"),
                "score": hit.get("_score"),
                "text": hit["_source"].get("text"),
            }
            for hit in hits
        ]
        return {"count": len(results), "results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

