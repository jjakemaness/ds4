#!/bin/bash
# GLM-5.3-Flash Q4_K on a 256 GB M3 Ultra, ~34.8 tok/s decode.
# Requires: trueimage/ds4 branch ds41f-m3ultra-perf, built.
#           GLM-5.3-Flash-Q4_K.gguf converted with glm53_kda_q8_inplace.py.
BUILD="${BUILD:-$HOME/ds4-m3perf}"      # the PR #1090 build
G="${G:-$HOME/ds4/gguf}"
cd "$BUILD" || exit 1                    # shaders load from ./metal
exec env DS4_GLM_ENABLE_KDA_Q8_INPUTS=1 ./ds4-server \
  -m "$G/GLM-5.3-Flash-Q4_K.gguf" \
  --vision "$G/GLM-5.3-Flash-Vision-Encoder.gguf" \
  --power 100 \
  --ctx "${CTX:-65536}" \
  --host 127.0.0.1 --port "${PORT:-8003}" \
  "$@"
