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
• Ollama + Qwen3-VL(로컬 VLM) 사용
• 노트북 6종 + 실습 데이터 생성 스크립트 포함

---

# 폴더 구성
• `1. Prompt Basics.ipynb` — Caption/OCR/Reasoning/Structured Output 프롬프트 비교
• `2. Image Info Extraction.ipynb` — 상품 이미지 분석·객체 인식·속성 추출·JSON 출력
• `3. OCR vs VLM.ipynb` — 전통적 OCR vs VLM 직접 이해 비교
• `4. Document AI.ipynb` — PDF·계약서·영수증·표 분석
• `5. Chart Analysis.ipynb` — 막대·선그래프·대시보드 분석
• `6. UI Understanding.ipynb` — 웹페이지·Figma·UX 분석
• `vlm_utils.py` — Ollama 호출 공용 헬퍼
• `data/` — 실습 이미지 생성 스크립트 + 생성된 샘플 데이터

---

# 사전 준비
1. **Ollama 설치**: https://ollama.com/download
2. **Qwen3-VL 모델 다운로드** (실습은 4b 기준, GPU 없어도 동작)
   ```bash
   ollama pull qwen3-vl:4b
   ```
3. (3번 노트북 보조 비교용) 텍스트 LLM 1개 다운로드
   ```bash
   ollama pull gemma3:4b
   ```
4. **uv 설치**: https://docs.astral.sh/uv/getting-started/installation/

---

# 가상환경 설정 (uv)

```bash
cd "2. VLM Prompt Engineering/examples"
uv sync
```

• `uv sync`가 `pyproject.toml`/`uv.lock` 기준으로 `.venv`를 만들고
  필요한 패키지(ollama, pillow, matplotlib, jupyterlab 등)를 설치
• Ollama 서버는 별도 설치 프로그램이 백그라운드 서비스로 자동 실행
  (`ollama list`로 설치된 모델 확인 가능)

---

# JupyterLab 실행

```bash
uv run jupyter lab
```

• 브라우저가 자동으로 열리며 `http://localhost:8888/lab` 접속
• 커널 선택 시 **Python (.venv)** 커널 사용

---

# 노트북 실행 순서
1. `1. Prompt Basics.ipynb` — 프롬프트 작성법 기본기
2. `2. Image Info Extraction.ipynb`
3. `3. OCR vs VLM.ipynb`
   - `pytesseract`로 실제 OCR을 시도하되, Tesseract 엔진이 없으면
     예시 텍스트로 자동 대체 (설치: https://github.com/UB-Mannheim/tesseract/wiki)
   - `gemma3:4b`를 텍스트 전용 LLM으로 사용 (사전에 pull 필요)
4. `4. Document AI.ipynb` — `pymupdf`로 PDF 페이지를 이미지로 렌더링
5. `5. Chart Analysis.ipynb`
6. `6. UI Understanding.ipynb`

---

# 참고 / 트러블슈팅
• `ConnectionError` → Ollama 서비스가 꺼져 있음. `ollama serve` 또는
  Ollama 앱이 실행 중인지 확인
• 응답이 느림 → GPU가 없거나 VRAM이 부족한 경우 CPU로 동작해 느릴 수 있음.
  더 작은 모델(`qwen3-vl:2b`)로 교체 가능
• `format="json"` 응답이 스키마와 다르면 → 프롬프트에 Few-shot 예시를
  추가하거나, 실패 시 재요청(retry)하는 로직을 추가해볼 것
