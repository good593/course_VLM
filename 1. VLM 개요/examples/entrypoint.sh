#!/bin/sh
# RunPod Network Volumes typically mount over /workspace, which hides the
# course notebooks baked into the image at build time (COPY put them in
# /opt/course-materials instead, precisely so this script can restore them).
#
# -n (no-clobber) so a volume from a previous session that already has
# student edits doesn't get overwritten by the pristine copies.
cp -rn /opt/course-materials/. /workspace/

exec "$@"
