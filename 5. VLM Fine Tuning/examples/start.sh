#!/bin/bash
set -e

# RunPod은 /workspace에 영구 네트워크 볼륨을 마운트하므로, 이미지 빌드 시점에
# 넣어둔 파일(/app/examples)은 컨테이너 시작 시 해당 경로에서 가려진다.
# 그래서 실행 시점(entrypoint)에 /workspace로 코드를 복사한다.
# --ignore-existing으로 이미 존재하는 파일(학생이 수정한 노트북 등)은 덮어쓰지 않고,
# .venv는 /app에 이미 있고 PATH로 참조되므로 영구 볼륨에 중복 저장하지 않도록 제외한다.
mkdir -p /workspace/examples
rsync -a --ignore-existing --exclude='.venv' /app/examples/ /workspace/examples/

cd /workspace/examples
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
  --NotebookApp.token= --notebook-dir=/workspace
