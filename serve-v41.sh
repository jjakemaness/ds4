#!/usr/bin/env bash
# DeepSeek V4.1 Flash 1017 HTTP server for a 256 GB Apple Silicon Mac.
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
MODEL_DIR="${MODEL_DIR:-$BUILD_DIR/gguf}"
PORT="${PORT:-8002}"
MODEL="$MODEL_DIR/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf"
VISION="$MODEL_DIR/DeepSeek-V4.1-Flash-1017-vision-BF16.gguf"
DSPARK="$MODEL_DIR/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf"

for path in "$BUILD_DIR/ds4-server" "$MODEL" "$VISION"; do
  [ -e "$path" ] || { echo "Missing required file: $path" >&2; exit 1; }
done
if [ -z "${NOSPEC:-}" ]; then
  [ -e "$DSPARK" ] || { echo "Missing required file: $DSPARK" >&2; exit 1; }
fi

if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "A server is already running on port ${PORT}."
  exit 0
fi
if pgrep -f "ds4-agent|ds4-server|ds4 -m" >/dev/null; then
  echo "Another DwarfStar model is already running (only one can run at a time):"
  pgrep -fl "ds4-agent|ds4-server|ds4 -m" | cut -c1-100
  exit 1
fi

cd "$BUILD_DIR" # Metal shaders load relative to the build checkout.

# Keep residency and queue keepalive enabled. Disabling either caused long-run
# stalls on the tested macOS 27 configuration.
SPEC=(--dspark --mtp-model "$DSPARK")
[ -n "${NOSPEC:-}" ] && SPEC=()
KV_DIR="${KV_DIR:-$HOME/.ds4/server-kv-v41-1017}"
mkdir -p "$KV_DIR"

exec ./ds4-server \
  -m "$MODEL" \
  --vision "$VISION" \
  ${SPEC[@]+"${SPEC[@]}"} \
  --power 100 \
  --ctx "${CTX:-262144}" \
  --kv-disk-dir "$KV_DIR" \
  --kv-disk-space-mb "${KV_MB:-20480}" \
  --kv-cache-cold-max-tokens 200000 \
  --host 127.0.0.1 --port "$PORT" \
  "$@"
