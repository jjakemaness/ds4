#!/bin/bash
# Sweep GLM decode knobs on the converted Q8-KDA file. One load per config.
G=/Users/benjamin_netenclawhu/ds4/gguf
B=/Users/benjamin_netenclawhu/ds4-m3perf
LOG=/Users/benjamin_netenclawhu/ds4/logs/sweep
mkdir -p "$LOG"
run() { # $1=label  rest=env assignments
  local label="$1"; shift
  pkill -f ds4-server 2>/dev/null; sleep 6
  local lg="$LOG/$label.log"
  ( cd "$B" && env "$@" DS4_GLM_ENABLE_KDA_Q8_INPUTS=1 ./ds4-server \
      -m "$G/GLM-5.3-Flash-Q4_K.gguf" --vision "$G/GLM-5.3-Flash-Vision-Encoder.gguf" \
      --power 100 --ctx 65536 --host 127.0.0.1 --port 8003 > "$lg" 2>&1 & )
  for i in $(seq 1 120); do
    curl -s -o /dev/null -m 5 -w '%{http_code}' http://127.0.0.1:8003/v1/models 2>/dev/null | grep -q 200 && break
    sleep 10
  done
  for r in 1 2; do
    curl -s -o /dev/null -m 900 http://127.0.0.1:8003/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"glm","messages":[{"role":"user","content":"Write a detailed technical explanation of consensus algorithms in distributed systems."}],"max_tokens":400,"temperature":0.7}'
  done
  local tps
  tps=$(grep -oE "decoding chunk=[0-9.]+ t/s avg=[0-9.]+ t/s" "$lg" | tail -1 | grep -oE "avg=[0-9.]+" | cut -d= -f2)
  printf "  %-34s %s t/s\n" "$label" "${tps:-FAILED}"
}
echo "  === GLM Q8-KDA knob sweep ==="
run "baseline(nsg4)"                 DS4_DUMMY=1
run "q8_mv_nsg=2"                    DS4_METAL_Q8_MV_NSG=2
run "q8_mv_nsg=8"                    DS4_METAL_Q8_MV_NSG=8
run "split_block_rows=8"             DS4_GLM_DECODE_SPLIT_BLOCK_ROWS=8
run "split_block_rows=32"            DS4_GLM_DECODE_SPLIT_BLOCK_ROWS=32
echo "  === done ==="
