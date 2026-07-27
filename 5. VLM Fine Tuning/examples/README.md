# VLM Fine Tuning 실습

노트북 2, 3, 5는 **RunPod GPU 인스턴스**에서 실행하는 것을 전제로 합니다.
로컬 CPU 환경에서는 4bit 양자화 로딩, LoRA/QLoRA 학습이 정상 동작하지 않을 수
있습니다. 노트북 1, 4는 가중치 학습이 없어 **로컬 + Ollama**만으로 실행됩니다
(단, 인터넷 연결과 HuggingFace 로그인은 모든 노트북에 필요합니다 — 아래 데이터셋 참고).

## 환경 변수 (.env)
모든 노트북이 `.env`에서 `HF_TOKEN`, `HF_DATASET_REPO_ID`를 읽습니다. 시작 전에 복사해두세요.

```bash
cp .env.example .env
```

- `HF_TOKEN`: HuggingFace 토큰 (write 권한). https://huggingface.co/settings/tokens 에서 발급
- `HF_DATASET_REPO_ID`: 노트북 1이 업로드하고, 이후 노트북들이 다운로드할 데이터셋 repo ID
  (예: `your-username/chartqa-kor-vlm-lab`) — 본인 계정/조직 네임스페이스로 직접 정하세요.

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

```bash
# 이미지 빌드 (로컬 또는 RunPod 빌드 파이프라인에서)
docker build -t vlm-finetuning-lab .

# 로컬에 GPU가 있다면 실행 예시
docker run --gpus all -p 8888:8888 \
  --env-file .env \
  -v "$(pwd)/data:/workspace/examples/data" \
  -v "$(pwd)/outputs:/workspace/examples/outputs" \
  vlm-finetuning-lab
```

RunPod에서는 이 이미지를 Pod의 **Custom Container Image**로 등록하고, `.env`의 값들을
환경 변수로 등록하면 됩니다(ChartQA_KOR, Qwen3-VL 다운로드에 필요).
컨테이너는 Jupyter Lab을 `8888` 포트로 띄웁니다.

## 사전 준비 (RunPod Pod 기준 — 노트북 2, 3, 5)
- GPU: 24GB VRAM급 이상 권장 (예: RTX 4090, A5000)
- Base image: `Dockerfile` (runpod/pytorch 공식 템플릿 기반)

```bash
cp .env.example .env   # HF_TOKEN, HF_DATASET_REPO_ID 채워넣기
uv sync
huggingface-cli login   # Qwen3-VL 다운로드용
```

## 사전 준비 (로컬 기준 — 노트북 1, 4)
```bash
cp .env.example .env   # HF_TOKEN, HF_DATASET_REPO_ID 채워넣기
uv sync
huggingface-cli login   # 노트북 1: ChartQA_KOR 다운로드/업로드용
ollama pull qwen3-vl:4b-instruct   # 노트북 4
```

## 실습 순서
1. **1. VLM 학습 데이터 준비.ipynb** — ChartQA_KOR을 내려받아 train/val로 나누고, HuggingFace Hub(`HF_DATASET_REPO_ID`)에 업로드 *(로컬)*
2. **2. QLoRA Fine-tuning.ipynb** — Hub 데이터셋을 내려받아 4bit 양자화 + LoRA로 base VLM(Qwen3-VL-2B-Instruct)을 실제 학습 *(RunPod GPU)*
3. **3. Fine-tuning 전후 비교 평가.ipynb** — base 모델 vs QLoRA 모델 정량 비교 *(RunPod GPU)*
4. **4. VERA 판단 기준 학습.ipynb** — 가중치 학습 없이 자연어 판단 기준(guideline)을 스스로 개선하는 verbalized learning 실습 *(로컬)*
5. **5. Vision Adapter 학습.ipynb** — Projector(어댑터)만 학습하는 경량 적응 *(RunPod GPU)*

