#!/bin/bash
# DwarfStar's own terminal coding agent (loads the model itself -- no server needed).
#   scripts/agent-v41.sh --chdir ~/your/project
# Same notes as serve-v41.sh: DSpark on, memory settings left at their defaults.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
G="${MODEL_DIR:-$REPO/gguf}"
cd "$REPO" || exit 1        # shaders load from ./metal -- must run from the repo
SPEC=(--dspark --mtp-model "$G/DeepSeek-V4.1-Flash-DSpark-pr1073-mxfp4.gguf")
[ -n "${NOSPEC:-}" ] && SPEC=()
exec ./ds4-agent \
  -m "$G/DeepSeek-V4.1-Flash-Q2.gguf" \
  --vision "$G/DeepSeek-V4.1-Flash-Vision.gguf" \
  "${SPEC[@]}" --power 100 --ctx "${CTX:-262144}" "$@"
