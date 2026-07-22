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
• JupyterLab 도커 이미지로 실행
• RunPod GPU Pod 기준 안내
• 노트북 3종 + 실습 데이터 다운로드 포함

---

# 폴더 구성
• `1. OCR - CRNN.ipynb` — CRNN 기반 OCR 실습
• `2. Image2Text.ipynb` — BLIP/GIT 이미지 캡셔닝 실습
• `3. Text2Image.ipynb` — Text-to-Image GAN 실습
• `Dockerfile`, `.dockerignore` — 실습 환경 빌드 파일
• `data/` — 로컬 샘플 데이터(이미지에는 포함되지 않음, 아래 참고)

---

# 사전 준비
• Docker 설치 (RunPod Pod는 기본 제공)
• GPU 환경 권장 (torch가 CUDA 지원 wheel로 설치됨)
• **GitHub Personal Access Token** 준비
  – `1. OCR - CRNN.ipynb`가 `good593/course_VLM`(비공개) 저장소에서
    CRNN 데이터셋을 내려받을 때 필요
  – 저장소에 read 권한이 있는 토큰이면 충분

---

# 이미지 빌드

```bash
cd "1. VLM 개요/examples"
docker build -t <dockerhub-id>/course-vlm-jupyter .
```

• 빌드 컨텍스트는 반드시 `examples` 폴더
• `data/` 폴더는 `.dockerignore`로 제외되어 이미지에 포함되지 않음
• 노트북이 설치하는 라이브러리(torch, transformers, kagglehub 등)는
  uv로 이미지 빌드 시 미리 설치되어 있음
• `<dockerhub-id>`는 본인 Docker Hub 계정으로 교체

---

# Docker Hub에 푸시

```bash
docker login
docker push <dockerhub-id>/course-vlm-jupyter
```

• RunPod은 Docker Hub(혹은 다른 레지스트리)에 있는 이미지를 pull해서
  실행하므로, 로컬 빌드 이미지를 미리 push해둬야 함
• 태그를 바꿔 재빌드했다면 그 태그로 다시 push

---

# RunPod에서 실행
1. RunPod 콘솔 → **Pods → Deploy** → GPU 종류 선택
2. **Container Image**에 `<dockerhub-id>/course-vlm-jupyter` 입력
3. **Expose HTTP Ports**에 `8888` 추가
4. **Environment Variables**에 `GITHUB_TOKEN=<개인 액세스 토큰>` 등록
   (CRNN 노트북의 데이터 다운로드용)
5. Deploy 후 RunPod이 제공하는 `https://<pod-id>-8888.proxy.runpod.net`
   주소로 접속 → JupyterLab이 인증 없이 바로 열림

---

# (참고) 로컬에서 먼저 테스트하고 싶다면

```bash
docker run -d \
  --gpus all \
  -p 8888:8888 \
  -e GITHUB_TOKEN=<개인 액세스 토큰> \
  --name vlm-lecture \
  <dockerhub-id>/course-vlm-jupyter
```

• `http://localhost:8888/lab`로 접속 (토큰/비밀번호 없음)
• GPU가 없는 로컬 환경이면 `--gpus all` 옵션 제거(CPU로 동작, 느림)

---

# 노트북 실행 순서
1. `1. OCR - CRNN.ipynb`
   • 데이터 다운로드 셀 실행 시 `GITHUB_TOKEN`이 없으면
     안내 메시지와 함께 에러 발생 → 컨테이너 실행 시 토큰을 꼭 설정
2. `2. Image2Text.ipynb`
   • 라이브러리는 모두 사전 설치되어 있어 `pip install` 셀은 그대로
     둬도 무해(재설치만 될 뿐)
3. `3. Text2Image.ipynb`
   • `kagglehub`로 Kaggle 데이터셋을 받는 셀이 있어
     **별도로 Kaggle 인증정보**(`~/.kaggle/kaggle.json` 또는
     `KAGGLE_USERNAME`/`KAGGLE_KEY` 환경변수)가 필요

---

# 참고 / 트러블슈팅
• `RuntimeError: GITHUB_TOKEN ...` → 컨테이너에 토큰 환경변수 미설정
• 이미지 캡셔닝 셀에서 `pip install git+...BLIP.git`은 원래 죽은
  코드(실제로는 `transformers`의 BLIP 클래스만 사용) → 무시해도 됨
• `data/` 폴더가 비어 있어도 정상 — CRNN 노트북 실행 시 자동 다운로드
