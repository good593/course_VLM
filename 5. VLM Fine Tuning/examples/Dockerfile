# RunPod 공식 PyTorch 템플릿 기반 — 학생들에게 동일한 개발 환경(Python/CUDA/패키지 버전)을 제공.
# RunPod이 태그 네이밍을 변경하여 2.4.0-py3.11-cuda12.4 태그는 더 이상 존재하지 않음.
# 동일 스펙(PyTorch 2.4.0, CUDA 12.4.1, Ubuntu 22.04)의 신규 태그로 대체.
FROM runpod/pytorch:0.7.0-cu1241-torch240-ubuntu2204

RUN pip install --no-cache-dir uv

# RunPod은 /workspace에 영구 볼륨을 마운트해 컨테이너 시작 시 그 경로를 덮어쓴다.
# 그래서 빌드 산출물(코드 + venv)은 /workspace가 아닌 /app에 만들어두고,
# start.sh가 실행 시점에 /workspace로 복사한다.
WORKDIR /app/examples

# 의존성만 먼저 복사해 uv sync 레이어를 캐싱
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .
RUN chmod +x start.sh

ENV PATH="/app/examples/.venv/bin:${PATH}"

EXPOSE 8888

ENTRYPOINT ["/app/examples/start.sh"]
