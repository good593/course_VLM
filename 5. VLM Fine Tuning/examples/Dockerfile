# RunPod 공식 PyTorch 템플릿 기반 — 학생들에게 동일한 개발 환경(Python/CUDA/패키지 버전)을 제공.
# RunPod이 태그 네이밍을 변경하여 2.4.0-py3.11-cuda12.4 태그는 더 이상 존재하지 않음.
# 동일 스펙(PyTorch 2.4.0, CUDA 12.4.1, Ubuntu 22.04)의 신규 태그로 대체.
FROM runpod/pytorch:0.7.0-cu1241-torch240-ubuntu2204

RUN pip install --no-cache-dir uv

# venv는 /app에 빌드 — /workspace가 아니어야 RunPod 네트워크 볼륨 마운트에
# 가려지지 않는다("1. VLM 개요/examples/Dockerfile"과 동일한 패턴).
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
ENV PATH="/app/.venv/bin:${PATH}"

# Ollama — 노트북 4(VERA)가 판단 기준 검증에 쓰는 로컬 VLM 서버.
# curl은 install.sh를 받는 데 필요 — 베이스 이미지에 이미 있어도 apt는 멱등이라 안전.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://ollama.com/install.sh | sh

# 모델도 빌드 시점에 이미지 안에 미리 받아둔다(pod이 뜰 때마다 다시 받지 않도록).
# OLLAMA_MODELS를 /workspace 밖(/opt)에 두는 이유는 위 /app과 동일 — RunPod 네트워크
# 볼륨이 /workspace를 덮어써서, 그 안에 두면 pod 시작마다 모델이 사라진다.
ENV OLLAMA_MODELS=/opt/ollama-models
RUN ollama serve & \
    OLLAMA_PID=$! && \
    for i in $(seq 1 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done && \
    ollama pull qwen3-vl:4b-instruct && \
    kill $OLLAMA_PID

# 코스 자료는 /opt/course-materials에 스테이징해두고, entrypoint.sh가
# 컨테이너 시작 시(볼륨 마운트 이후) /workspace로 복원한다.
# .venv는 .dockerignore로 제외되어 여기 포함되지 않는다.
#
# entrypoint.sh도 이 COPY에 함께 딸려오므로(빌드 컨텍스트의 일부), 복원 대상에서
# 빠지도록 course-materials 밖으로 옮긴다.
COPY . /opt/course-materials/
RUN mv /opt/course-materials/entrypoint.sh /opt/entrypoint.sh && \
    chmod +x /opt/entrypoint.sh

WORKDIR /workspace

EXPOSE 8888

ENTRYPOINT ["/opt/entrypoint.sh"]

# 토큰/비밀번호 없음(요구사항) — 공개 인터넷이 아닌 개인 RunPod Pod에서만
# 안전함. allow_origin=* + allow_remote_access는 RunPod 리버스 프록시 뒤에서
# CORS/Host-check 403을 피하기 위함.
# allow_hidden=True는 학생이 채워야 하는 .env/.env.example이 파일 브라우저에서
# 보이게 하기 위함(점(.)으로 시작하는 파일은 기본적으로 서버가 목록에서 제외함).
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=", \
     "--ServerApp.password=", \
     "--ServerApp.allow_origin=*", \
     "--ServerApp.allow_remote_access=True", \
     "--ContentsManager.allow_hidden=True"]
