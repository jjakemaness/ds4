#!/bin/bash
# Median-of-3 confirmation for the top GLM configs.
G=/Users/benjamin_netenclawhu/ds4/gguf
B=/Users/benjamin_netenclawhu/ds4-m3perf
LOG=/Users/benjamin_netenclawhu/ds4/logs/confirm; mkdir -p "$LOG"
one() { # $1=label rest=env -> prints 3 samples + median
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
  local vals=()
  for r in 1 2 3; do
    : > "$lg.r$r"
    curl -s -o /dev/null -m 900 http://127.0.0.1:8003/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"glm","messages":[{"role":"user","content":"Write a detailed technical explanation of consensus algorithms in distributed systems."}],"max_tokens":400,"temperature":0.7}'
    vals+=("$(grep -oE 'decoding chunk=[0-9.]+ t/s avg=[0-9.]+ t/s' "$lg" | tail -1 | grep -oE 'avg=[0-9.]+' | cut -d= -f2)")
  done
  local med; med=$(printf '%s\n' "${vals[@]}" | sort -n | sed -n 2p)
  printf "  %-26s  %-7s %-7s %-7s  median %s\n" "$label" "${vals[0]}" "${vals[1]}" "${vals[2]}" "$med"
}
echo "  === median-of-3 confirmation ==="
one "default"          DS4_DUMMY=1
one "split_rows=4"     DS4_GLM_DECODE_SPLIT_BLOCK_ROWS=4
one "split_rows=12"    DS4_GLM_DECODE_SPLIT_BLOCK_ROWS=12
one "split_rows=64"    DS4_GLM_DECODE_SPLIT_BLOCK_ROWS=64
echo "  === done ==="
