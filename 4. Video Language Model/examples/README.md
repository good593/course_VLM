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
• Ollama + Qwen3-VL(로컬 VLM) + Qwen3-Embedding(로컬 임베딩) 사용
• 노트북 4종 + 실습 영상(`data/`) 포함

---

# 폴더 구성
• `1. Frame 추출과 전처리.ipynb` — OpenCV 메타데이터/균등·고정fps·장면전환 샘플링 비교
• `2. 영상 요약과 이벤트 탐지.ipynb` — 전체 요약 + 타임라인 JSON 이벤트 탐지
• `3. 행동 분석.ipynb` — 정상/이상행동 판별 (CCTV 스타일)
• `4. Video RAG 파이프라인.ipynb` — 긴 영상 클립 분할 + Multi-Vector 검색·질의응답
• `video_utils.py` — 영상 처리 + Ollama 호출 공용 헬퍼
• `data/` — 실습 영상 (출처/라이선스는 `data/README.md` 참고)

---

# 사전 준비
1. **Ollama 설치**: https://ollama.com/download
2. **모델 다운로드**
   ```bash
   ollama pull qwen3-vl:4b          # 프레임 이해 (전 노트북 공통)
   ollama pull qwen3-embedding:0.6b # 4번 노트북 Video RAG 전용
   ```
3. **uv 설치**: https://docs.astral.sh/uv/getting-started/installation/

---

# 가상환경 설정 (uv)

```bash
cd "4. Video Language Model/examples"
uv sync
```

• `uv sync`가 `pyproject.toml`/`uv.lock` 기준으로 `.venv`를 만들고
  필요한 패키지(opencv-python, ollama, chromadb, jupyterlab 등)를 설치
• 실습 영상은 `data/`에 이미 포함되어 있어 별도 다운로드 불필요

---

# JupyterLab 실행

```bash
uv run jupyter lab
```

• 브라우저가 자동으로 열리며 `http://localhost:8888/lab` 접속
• 커널 선택 시 **Python (.venv)** 커널 사용

---

# 노트북 실행 순서
1. `1. Frame 추출과 전처리.ipynb` — 프레임 추출 전략 3가지 비교 (기초)
2. `2. 영상 요약과 이벤트 탐지.ipynb`
3. `3. 행동 분석.ipynb`
4. `4. Video RAG 파이프라인.ipynb` — 1~3번보다 실행 시간이 길 수 있음 (클립 수만큼 VLM 호출)

---

# 참고 / 트러블슈팅
• `ConnectionError` → Ollama 서비스가 꺼져 있음. `ollama serve` 또는
  Ollama 앱이 실행 중인지 확인
• 응답이 느림 → 프레임 수를 줄이거나(`n_frames`), 해상도를 낮추면(`max_width`)
  속도가 빨라짐. VLM 호출 1번에 프레임이 많을수록 느려짐
• 장면전환 감지가 프레임을 너무 적게/많이 뽑으면 → `threshold` 값을 조정
  (값이 클수록 큰 변화만 감지, 작을수록 민감하게 감지)
• `format="json"` 응답이 스키마와 다르면 → 프롬프트에 스키마 예시를 더 구체적으로
  추가하거나 재요청(retry) 로직 추가
