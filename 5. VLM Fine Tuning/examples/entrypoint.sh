#!/bin/sh
# RunPod Network Volumes typically mount over /workspace, which hides the
# course notebooks baked into the image at build time (COPY put them in
# /opt/course-materials instead, precisely so this script can restore them).
#
# -n (no-clobber) so a volume from a previous session that already has
# student edits doesn't get overwritten by the pristine copies.
cp -rn /opt/course-materials/. /workspace/

# 노트북 4(VERA)가 붙을 수 있도록 Ollama 서버를 백그라운드로 띄운다.
# 모델(qwen3-vl:4b-instruct)은 빌드 시점에 이미 OLLAMA_MODELS(/opt/ollama-models,
# Dockerfile 참고)에 구워져 있어 pod 시작 시 다시 받지 않는다.
ollama serve > /var/log/ollama.log 2>&1 &

exec "$@"
