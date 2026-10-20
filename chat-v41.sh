#!/usr/bin/env bash
# DeepSeek V4.1 Flash 1017 native DwarfStar coding agent.
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
MODEL_DIR="${MODEL_DIR:-$BUILD_DIR/gguf}"
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
MODEL="$MODEL_DIR/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf"
VISION="$MODEL_DIR/DeepSeek-V4.1-Flash-1017-vision-BF16.gguf"
DSPARK="$MODEL_DIR/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf"

for path in "$BUILD_DIR/ds4-agent" "$MODEL" "$VISION"; do
  [ -e "$path" ] || { echo "Missing required file: $path" >&2; exit 1; }
done
if [ -z "${NOSPEC:-}" ]; then
  [ -e "$DSPARK" ] || { echo "Missing required file: $DSPARK" >&2; exit 1; }
fi

if pgrep -f "ds4-agent|ds4-server|ds4 -m" >/dev/null; then
  echo "Another DwarfStar model is already running (only one can run at a time):"
  pgrep -fl "ds4-agent|ds4-server|ds4 -m" | cut -c1-100
  exit 1
fi

cd "$BUILD_DIR" # Metal shaders load relative to the build checkout.

# Keep residency and queue keepalive enabled. The native agent owns inference
# directly and therefore does not run beside the HTTP server.
SPEC=(--dspark --mtp-model "$DSPARK")
[ -n "${NOSPEC:-}" ] && SPEC=()

exec ./ds4-agent \
  -m "$MODEL" \
  --vision "$VISION" \
  ${SPEC[@]+"${SPEC[@]}"} \
  --power 100 \
  --ctx "${CTX:-262144}" \
  --chdir "$PROJECT_DIR" \
  "$@"
