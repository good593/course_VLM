# JupyterLab image for the course_VLM lecture materials (1. VLM 개요/examples).
# Built for GPU pods (e.g. RunPod) — Linux torch wheels bundle CUDA runtime libs,
# so no dedicated nvidia/cuda base image is required; the host's NVIDIA driver
# is what actually needs to be present (RunPod GPU pods provide this).
FROM python:3.11-slim-bookworm

# git      -> general convenience for students working inside the container
# curl     -> fetch the uv installer / static binary
# build-essential -> native extensions some deps compile on install
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (static binary) from its official distroless image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# Pre-install every package the notebooks themselves `!pip install` at runtime,
# so the in-notebook install cells become no-ops. Versions mirror what each
# notebook pins; "pip" itself is included so those redundant install cells
# don't fail on a missing pip module.
#
#  1. OCR - CRNN.ipynb      -> tqdm, matplotlib, seaborn
#  2. Image2Text.ipynb      -> transformers, torch, torchvision, typing_extensions,
#                               pillow, requests, koreanize-matplotlib
#                               (its `pip install git+...salesforce/BLIP.git` cell is
#                               dead code — the notebook only ever uses transformers'
#                               own BlipProcessor/BlipForConditionalGeneration, and the
#                               BLIP repo has no setup.py/pyproject.toml to pip-install
#                               anyway, so it's intentionally not pre-installed here)
#  3. Text2Image.ipynb      -> imageio, h5py, torch, torchvision, kagglehub
RUN uv pip install --system --no-cache \
    pip \
    jupyterlab \
    tqdm \
    matplotlib \
    seaborn \
    pillow \
    requests \
    koreanize-matplotlib \
    typing_extensions==4.15.0 \
    torch==2.8.0 \
    torchvision==0.23.0 \
    transformers==4.56.2 \
    kagglehub==1.0.1 \
    imageio==2.37.0 \
    h5py==3.14.0

# Course notebooks, minus data/ (see .dockerignore) — that dataset is
# fetched at runtime instead (see "1. OCR - CRNN.ipynb", data-download cell).
# Build context is this "examples" folder itself.
COPY . /workspace/

EXPOSE 8888

# No auth token/password (per requirements) — only safe because this runs on a
# private RunPod pod, not exposed directly to the public internet.
# allow_origin='*' + allow_remote_access avoid CORS/Host-check 403s behind
# RunPod's reverse proxy.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=", \
     "--ServerApp.password=", \
     "--ServerApp.allow_origin=*", \
     "--ServerApp.allow_remote_access=True"]
