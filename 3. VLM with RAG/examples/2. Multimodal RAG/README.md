---
style: |
  img {
    display: block;
    float: none;
    margin-left: auto;
    margin-right: auto;
  }
marp: true
paginate: true
---

# Examples 실습 환경
• uv 가상환경으로 로컬 실행
• OpenAI `gpt-4o-mini`(비전 요약/QA) + `text-embedding-3-small`(텍스트 임베딩) 사용
• Chroma로 구현하는 Multi-Vector Retriever 패턴 노트북 2종

---

# 폴더 구성
• `1. Multi-Vector Retriever.ipynb` — 텍스트/표/차트/이미지 혼합 문서에
  Multi-Vector Retriever 패턴(요약 임베딩 + 원본 검색) 적용
• `2. PDF & Table QA.ipynb` — PDF 페이지 이미지 검색·QA, 표 질의응답,
  이미지 비교 질의(QA)
• `rag_utils.py` — OpenAI 호출(VLM 요약/QA, 텍스트 임베딩) 공용 헬퍼
• `data/` — 실습 데이터 (아래 "데이터 출처" 참고) + 생성 스크립트

---

# 데이터 출처
표·차트·PDF는 **국가데이터처(통계청)가 실제로 발표한 공공데이터**입니다.

> 「2026년 5월 온라인쇼핑동향」 보도자료 (2026-07-01), 국가데이터처(통계청)
> https://mods.go.kr/board.es?mid=a10301120300&bid=241&act=view&list_no=445708
> 라이선스: **공공누리 제1유형(출처표시)** — https://www.kogl.or.kr

원본은 44페이지이며, 이 중 5개 페이지만 발췌해 `data/online_shopping_trend.pdf`로
만들고 표/차트는 이미지로도 잘라 저장했습니다 (`data/fetch_real_report.py`가 담당,
산출물은 이미 커밋되어 있어 재실행이 필수는 아닙니다). 제품 이미지 3종(이어폰/
스마트워치/스피커)만 이미지 QA 실습용으로 코드로 그린 아이콘입니다.

---

# 사전 준비
1. **OpenAI API 키 발급**: https://platform.openai.com/api-keys
2. 프로젝트 폴더에 `.env` 파일 생성 (`.env.example` 복사 후 값 채우기)
   ```
   OPENAI_API_KEY="sk-..."
   ```
3. **uv 설치**: https://docs.astral.sh/uv/getting-started/installation/

> 이 노트북들은 API 호출당 비용이 발생합니다(호출 수십 회 수준으로 소액).
> 키는 `.env`에만 두고 커밋하지 마세요 (`.gitignore`에 이미 포함됨).

---

# 가상환경 설정 (uv)

```bash
cd "3. VLM with RAG/examples/2. Multimodal RAG"
uv sync
```

• `uv sync`가 `pyproject.toml`/`uv.lock` 기준으로 `.venv`를 만들고
  필요한 패키지(chromadb, openai, pymupdf, jupyter 등)를 설치

---

# 실습 데이터 생성
표/차트/PDF(실제 공공데이터)는 이미 `data/`에 커밋되어 있어 별도 실행이 필요 없습니다.
제품 이미지 3종만 코드로 생성하면 됩니다.

```bash
uv run python data/generate_sample_data.py
```

• 실행하면 `data/product_*.png` 3종이 생성됨
• 표/차트/PDF를 새로고침하고 싶다면 `uv run python data/fetch_real_report.py`
  (인터넷 연결 필요, 통계청 서버에서 다시 내려받음)

---

# JupyterLab 실행

```bash
uv run jupyter lab
```

• 브라우저가 자동으로 열리며 `http://localhost:8888/lab` 접속
• 커널 선택 시 **Python (.venv)** 커널 사용

---

# 노트북 실행 순서
1. `1. Multi-Vector Retriever.ipynb`
   - 텍스트 1개 + 표/차트/제품 이미지 5개로 Multi-Vector Retriever 구축
   - API 호출 약 8회 (이미지 요약 5회 + 최종 답변 생성 2회 + 임베딩)
2. `2. PDF & Table QA.ipynb`
   - PDF 5페이지를 이미지로 렌더링 후 1번과 동일한 패턴으로 색인
   - PDF QA(페이지 인용) → 표 QA(마크다운+원본 병행) → 이미지 비교 QA
   - 실제 공공데이터라 24개 상품군이 담긴 조밀한 차트 등 난이도가 있는 질문도
     포함되어 있음 — VLM이 이미지만으로 정확히 못 읽는 사례를 의도적으로 보여줌
   - API 호출 약 9회

---

# 참고 / 트러블슈팅
• `AuthenticationError` → `.env`의 `OPENAI_API_KEY`가 비어있거나 잘못됨.
  `rag_utils.py`가 `load_dotenv()`로 `.env`를 읽으므로 파일 위치를 확인
• `RateLimitError` → 짧은 시간에 호출이 몰린 경우. 잠시 후 재시도
• 검색이 엉뚱한 문서를 가져오면 → 요약 프롬프트를 더 구체적으로(수치·색상·
  형태 등) 바꿔볼 것 — 검색 품질은 전적으로 요약 텍스트 품질에 좌우됨
• `chroma_db/` 폴더는 노트북을 다시 실행하면 컬렉션을 지우고 새로 만들므로
  삭제해도 안전함
