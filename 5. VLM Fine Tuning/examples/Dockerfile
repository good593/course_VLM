# RunPod 공식 PyTorch 템플릿 기반 — 학생들에게 동일한 개발 환경(Python/CUDA/패키지 버전)을 제공.
# RunPod이 태그 네이밍을 변경하여 2.4.0-py3.11-cuda12.4 태그는 더 이상 존재하지 않음.
# 동일 스펙(PyTorch 2.4.0, CUDA 12.4.1, Ubuntu 22.04)의 신규 태그로 대체.
FROM runpod/pytorch:0.7.0-cu1241-torch240-ubuntu2204

WORKDIR /workspace/examples

RUN pip install --no-cache-dir uv

# 의존성만 먼저 복사해 uv sync 레이어를 캐싱
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

ENV PATH="/workspace/examples/.venv/bin:${PATH}"

EXPOSE 8888

CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token="]
