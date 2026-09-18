#!/bin/bash
# DeepSeek V4.1 Flash Q2 server on port 8002 (OpenAI-compatible: http://127.0.0.1:8002/v1)
# with DSpark speculative decoding. Ctrl-C to stop.
#
#   scripts/serve-v41.sh              start the server
#   NOSPEC=1 scripts/serve-v41.sh     without DSpark (steady ~38 t/s, byte-identical output)
#   CTX=524288 scripts/serve-v41.sh   bigger context window
#   MODEL_DIR=/path/to/gguf ...       where the GGUF files live (default: ./gguf)
#
# DO NOT set DS4_METAL_NO_RESIDENCY or DS4_METAL_DISABLE_QUEUE_KEEPALIVE: on macOS 27
# they cause multi-minute freezes on long prompts. See V41-MACOS27.md.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
G="${MODEL_DIR:-$REPO/gguf}"
PORT="${PORT:-8002}"
if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "A server is already running on port ${PORT}."; exit 0
fi
if pgrep -f "ds4-agent|ds4-server|ds4 -m" >/dev/null; then
  echo "Another DwarfStar model is already running (only one can run at a time)."; exit 1
fi
cd "$REPO" || exit 1        # shaders load from ./metal -- must run from the repo
SPEC=(--dspark --mtp-model "$G/DeepSeek-V4.1-Flash-DSpark-pr1073-mxfp4.gguf")
[ -n "${NOSPEC:-}" ] && SPEC=()
KV_DIR="${KV_DIR:-$HOME/.ds4/server-kv-v41}"; mkdir -p "$KV_DIR"
exec ./ds4-server \
  -m "$G/DeepSeek-V4.1-Flash-Q2.gguf" \
  --vision "$G/DeepSeek-V4.1-Flash-Vision.gguf" \
  "${SPEC[@]}" \
  --power 100 --ctx "${CTX:-262144}" \
  --kv-disk-dir "$KV_DIR" --kv-disk-space-mb "${KV_MB:-20480}" \
  --kv-cache-cold-max-tokens 200000 \
  --host 127.0.0.1 --port "$PORT" \
  "$@"
