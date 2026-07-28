# VLM Fine Tuning 실습

노트북 2, 3, 5는 **RunPod GPU 인스턴스**에서 실행하는 것을 전제로 합니다.
로컬 CPU 환경에서는 4bit 양자화 로딩, LoRA/QLoRA 학습이 정상 동작하지 않을 수
있습니다. 노트북 1, 4는 가중치 학습이 없어 **Ollama**만으로 실행됩니다
(단, 인터넷 연결과 HuggingFace 로그인은 모든 노트북에 필요합니다 — 아래 데이터셋 참고).

- 이 저장소의 `Dockerfile`로 빌드한 RunPod Pod에는 Ollama와 모델(`qwen3-vl:4b-instruct`)이
  이미 설치되어 있어, 노트북 4도 그 Pod 안에서 바로 실행됩니다.
- Docker 없이 로컬에서 노트북 4를 돌리려면 [Ollama](https://ollama.com)를 직접 설치하고
  `ollama pull qwen3-vl:4b-instruct`로 모델을 받아둬야 합니다.

## 환경 변수 (.env)
모든 노트북이 `.env`에서 `HF_TOKEN`, `HF_DATASET_REPO_ID`를 읽습니다. 시작 전에 복사해두세요.

```bash
cp .env.example .env
```

- `HF_TOKEN`: HuggingFace 토큰 (write 권한). https://huggingface.co/settings/tokens 에서 발급
- `HF_DATASET_REPO_ID`: 노트북 1이 업로드하고, 이후 노트북들이 다운로드할 데이터셋 repo ID
  (예: `your-username/chartqa-kor-vlm-lab`) — 본인 계정/조직 네임스페이스로 직접 정하세요.

> JupyterLab(RunPod Pod)에서는 `.env`/`.env.example`처럼 `.`으로 시작하는 파일이
> 파일 브라우저에 기본적으로 숨겨져 있습니다. 파일 브라우저 빈 공간에서
> 우클릭 → **Show Hidden Files**를 켜면 보입니다.

## 데이터셋: ChartQA_KOR
자체 생성 데이터 대신 실제 공개 데이터셋인
**[Tutoruslabs/ChartQA_KOR](https://huggingface.co/datasets/Tutoruslabs/ChartQA_KOR)**
을 사용합니다. 차트 이미지에 대한 한국어 질의응답 데이터셋입니다.

- **gated 데이터셋**입니다 — 데이터셋 페이지에서 로그인 후 이용약관에 동의해야 다운로드할 수 있습니다.
- **평가(benchmark)용으로 test split(2,000개)만** 제공됩니다.
  `1. VLM 학습 데이터 준비.ipynb`가 이를 train(80%)/val(20%)로 나눈 뒤,
  **`HF_DATASET_REPO_ID`에 업로드**합니다.
- 이후 노트북들(2, 3, 4, 5)은 원본 ChartQA_KOR이 아니라 **`HF_DATASET_REPO_ID`에서
  다운로드**해서 사용합니다 — 로컬에 `data/` 폴더가 없는 새 환경(예: RunPod Pod)에서도
  노트북 1을 다시 실행할 필요 없이 바로 실습할 수 있습니다.
- 정답 채점은 ChartQA 공식 방식인 **relaxed accuracy**(숫자 답은 5% 오차까지 허용,
  텍스트 답은 정규화 후 완전 일치)를 `vlm_finetune_utils.judge_answer`로 구현해 사용합니다.

## Docker로 개발 환경 통일하기
학생들이 동일한 환경(파이썬/CUDA/패키지 버전)에서 실습할 수 있도록 Dockerfile을 제공합니다.
RunPod Pod의 컨테이너 이미지로 그대로 사용할 수 있습니다.

RunPod은 Pod 시작 시 **`/workspace`에 영구 네트워크 볼륨을 마운트**하며, 이 마운트가
이미지 빌드 시점에 그 경로에 넣어둔 내용을 덮어씁니다. 그래서 이 Dockerfile은 코드와
venv를 `/workspace`가 아닌 `/app`, `/opt/course-materials`에 빌드해두고, 컨테이너가
시작될 때 `entrypoint.sh`가 `/workspace`로 복사한 뒤 Jupyter Lab을 띄웁니다
(`1. VLM 개요/examples/Dockerfile`과 동일한 패턴). 이미 존재하는 파일(학생이 수정한
노트북 등)은 덮어쓰지 않으므로, Pod을 재시작해도 작업 내용이 보존됩니다.

```bash
# 이미지 빌드 (로컬 또는 RunPod 빌드 파이프라인에서)
docker build -t <dockerhub-id>/vlm-finetuning-lab .

# 로컬에 GPU가 있다면 실행 예시
docker run --gpus all -p 8888:8888 \
  --env-file .env \
  -v "$(pwd)/data:/workspace/data" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  <dockerhub-id>/vlm-finetuning-lab
```

RunPod에서는 이 이미지를 Pod의 **Custom Container Image**로 등록하고, `.env`의 값들을
환경 변수로 등록하면 됩니다(ChartQA_KOR, Qwen3-VL 다운로드에 필요).
컨테이너는 Jupyter Lab을 `8888` 포트로 띄웁니다(토큰/비밀번호 없음).

