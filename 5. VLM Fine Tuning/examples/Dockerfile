# RunPod 공식 PyTorch 템플릿 기반 — 학생들에게 동일한 개발 환경(Python/CUDA/패키지 버전)을 제공.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4

WORKDIR /workspace/examples

RUN pip install --no-cache-dir uv

# 의존성만 먼저 복사해 uv sync 레이어를 캐싱
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

ENV PATH="/workspace/examples/.venv/bin:${PATH}"

EXPOSE 8888

CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token="]
