# VLM Fine Tuning 실습

노트북 2, 3, 5는 **RunPod GPU 인스턴스**에서 실행하는 것을 전제로 합니다.
로컬 CPU 환경에서는 4bit 양자화 로딩, LoRA/QLoRA 학습이 정상 동작하지 않을 수
있습니다. 노트북 1, 4는 가중치 학습이 없어 **로컬 + Ollama**만으로 실행됩니다.

## 사전 준비 (RunPod Pod 기준 — 노트북 2, 3, 5)
- GPU: 24GB VRAM급 이상 권장 (예: RTX 4090, A5000)
- Base image: PyTorch + CUDA가 포함된 RunPod 템플릿 (예: `runpod/pytorch:2.4.0-py3.11-cuda12.4`)

```bash
uv sync
```

## 사전 준비 (로컬 기준 — 노트북 1, 4)
```bash
uv sync
ollama pull qwen3-vl:4b
```

## 실습 순서
1. **1. VLM 학습 데이터 준비.ipynb** — instruction-following 포맷의 장난감 "제품 불량 검사" 데이터셋 생성 *(로컬)*
2. **2. QLoRA Fine-tuning.ipynb** — 4bit 양자화 + LoRA로 base VLM(Qwen2-VL-2B-Instruct)을 실제 학습 *(RunPod GPU)*
3. **3. Fine-tuning 전후 비교 평가.ipynb** — base 모델 vs QLoRA 모델 정량 비교 *(RunPod GPU)*
4. **4. VERA 판단 기준 학습.ipynb** — 가중치 학습 없이 자연어 판단 기준(guideline)을 스스로 개선하는 verbalized learning 실습 *(로컬)*
5. **5. Vision Adapter 학습.ipynb** — Projector(어댑터)만 학습하는 경량 적응 *(RunPod GPU)*

## 데이터셋에 대한 안내
실제 서비스 데이터 대신 `PIL`로 즉석에서 생성하는 **장난감(toy) 데이터셋**을 사용합니다.
"사각형 제품 이미지 + 모서리의 스크래치 유무"라는 단순한 패턴이라, 학습이 실제로
효과가 있는지(전후 비교) 눈으로 바로 확인할 수 있습니다. 실무에서는 이 부분을
직접 라벨링한 도메인 데이터로 교체하면 됩니다.

## 참고
- 슬라이드: `../VLM Fine Tuning.md`
- 공통 유틸리티: `vlm_finetune_utils.py`
