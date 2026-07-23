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

# Install uv (static binary) from its official distroless image.
# Pinned (not :latest) so every student's image is built with the same uv.
COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /usr/local/bin/uv

WORKDIR /workspace

# Pre-install every package the notebooks themselves `!pip install` at runtime,
# so the in-notebook install cells become no-ops. Versions mirror what each
# notebook pins; "pip" itself is included so those redundant install cells
# don't fail on a missing pip module.
#
# Every version below is pinned (none left floating) so every student builds
# the exact same environment instead of whatever happens to be latest on the
# day they build the image.
#
#  1. OCR - CRNN.ipynb      -> tqdm, matplotlib, seaborn
#                               (tqdm.notebook needs ipywidgets to render its
#                               progress bar in JupyterLab, or it raises
#                               "ImportError: IProgress not found")
#  2. Image2Text.ipynb      -> transformers, torch, torchvision, typing_extensions,
#                               pillow, requests, koreanize-matplotlib
#                               (its `pip install git+...salesforce/BLIP.git` cell is
#                               dead code — the notebook only ever uses transformers'
#                               own BlipProcessor/BlipForConditionalGeneration, and the
#                               BLIP repo has no setup.py/pyproject.toml to pip-install
#                               anyway, so it's intentionally not pre-installed here)
#  3. Text2Image.ipynb      -> imageio, h5py, torch, torchvision, kagglehub,
#                               diffusers, accelerate, xformers, huggingface_hub
#                               (its Stable Diffusion section previously left
#                               huggingface_hub unpinned in-notebook; pinned here
#                               to the newest 0.x release compatible with
#                               transformers/diffusers/accelerate's own
#                               huggingface-hub>=0.34.0,<1.0-style constraints)
RUN uv pip install --system --no-cache \
    pip==26.1.2 \
    jupyterlab==4.6.1 \
    tqdm==4.69.0 \
    ipywidgets==8.1.8 \
    matplotlib==3.11.1 \
    seaborn==0.13.2 \
    pillow==12.3.0 \
    requests==2.34.2 \
    koreanize-matplotlib==0.1.1 \
    typing_extensions==4.15.0 \
    torch==2.8.0 \
    torchvision==0.23.0 \
    transformers==4.56.2 \
    kagglehub==1.0.1 \
    imageio==2.37.0 \
    h5py==3.14.0 \
    huggingface_hub==0.34.6 \
    diffusers==0.35.1 \
    accelerate==1.10.1 \
    xformers==0.0.32.post2

# Course notebooks, minus data/ (see .dockerignore) — that dataset is
# fetched at runtime instead (see "1. OCR - CRNN.ipynb", data-download cell).
# Build context is this "examples" folder itself.
#
# Staged outside /workspace on purpose: RunPod Network Volumes are commonly
# mounted at /workspace, which would otherwise hide these files behind an
# empty volume. entrypoint.sh restores them into /workspace at container
# startup (after the volume is already mounted), without clobbering any
# student edits a previous session already saved there.
#
# entrypoint.sh rides along in this COPY (it's part of the build context
# too), then gets moved out of course-materials so it isn't itself among the
# files restored into /workspace at runtime.
COPY . /opt/course-materials/
RUN mv /opt/course-materials/entrypoint.sh /opt/entrypoint.sh && \
    chmod +x /opt/entrypoint.sh

EXPOSE 8888

ENTRYPOINT ["/opt/entrypoint.sh"]

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
