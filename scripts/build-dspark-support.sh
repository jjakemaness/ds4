#!/bin/bash
# Build the DSpark support file for DeepSeek-V4.1-Flash-Q2.gguf WITHOUT downloading
# the whole ~330 GB checkpoint.
#
# The converter plans the full model even for a DSpark-only output, so it wants all
# 48 shards present -- but it only READS shards 44-46. This downloads those three
# (~8 GB), fetches just the header (table of contents) of the other 45, and makes
# header-only placeholder files truncated to full size (sparse: ~10 MB on disk).
#
#   scripts/build-dspark-support.sh            convert (~8 GB download, ~1 min)
#   DRY_RUN=1 scripts/build-dspark-support.sh  check everything without writing it
#
# Needs: hf (huggingface-cli), uv, curl, python3. The revision must match the one
# your target GGUF was built from (the ds41f-q2 download uses this one).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REV="${REV:-df42c109f1defefcbfcedbe7d905718a12266e40}"
HF_REPO="deepseek-ai/DeepSeek-V4.1-Flash"
SRC="${SRC:-$REPO/dspark-source}"      # real files (~8 GB)
STUB="${STUB:-$REPO/dspark-stub}"      # placeholders + links to the real files
OUT="${OUT:-${MODEL_DIR:-$REPO/gguf}/DeepSeek-V4.1-Flash-DSpark-pr1073-mxfp4.gguf}"
REAL=(model-00044-of-00048.safetensors model-00045-of-00048.safetensors model-00046-of-00048.safetensors)

echo "1/4 downloading config, tokenizer, index and shards 44-46 (~8 GB)..."
hf download "$HF_REPO" --revision "$REV" --local-dir "$SRC" \
  config.json inference/config.json model.safetensors.index.json tokenizer.json "${REAL[@]}"

echo "2/4 making header-only placeholders for the other 45 shards..."
mkdir -p "$STUB/inference"
for f in config.json tokenizer.json model.safetensors.index.json "${REAL[@]}"; do ln -sf "$SRC/$f" "$STUB/$f"; done
ln -sf "$SRC/inference/config.json" "$STUB/inference/config.json"
curl -fsSL "https://huggingface.co/api/models/$HF_REPO/tree/$REV?recursive=1" | python3 -c '
import json,sys
for f in json.load(sys.stdin):
    p=f.get("path","")
    if p.startswith("model-") and p.endswith(".safetensors"):
        print(p,(f.get("lfs") or {}).get("size") or f.get("size"))' > "$STUB/.sizes"
while read -r name size; do
  [ -e "$STUB/$name" ] && continue
  url="https://huggingface.co/$HF_REPO/resolve/$REV/$name"
  curl -fsSL -r 0-7 "$url" -o "$STUB/.len"
  n=$(python3 -c "import struct,sys;print(struct.unpack('<Q',open(sys.argv[1],'rb').read(8))[0])" "$STUB/.len")
  curl -fsSL -r "0-$((n+7))" "$url" -o "$STUB/$name.part"
  [ "$(stat -f %z "$STUB/$name.part")" -eq $((n+8)) ] || { echo "bad header for $name"; exit 1; }
  mv "$STUB/$name.part" "$STUB/$name"; truncate -s "$size" "$STUB/$name"
done < "$STUB/.sizes"
rm -f "$STUB/.len"
echo "   $(ls "$STUB"/model-*.safetensors | wc -l | tr -d ' ') shards present, $(du -sh "$STUB" | cut -f1) real disk"

echo "3/4 building the quantization library..."
make -C "$REPO/gguf-tools" libds4quants.dylib >/dev/null

echo "4/4 converting (MXFP4 drafter, lossless from the source FP4 experts)..."
ARGS=(--hf "$STUB" --source-revision "$REV" --quant mxfp4 --dspark-out "$OUT")
[ -n "${DRY_RUN:-}" ] && ARGS+=(--dry-run)
cd "$REPO" && uv run --with numpy --with tokenizers --with sympy \
  python gguf-tools/deepseek41_quantize.py "${ARGS[@]}" | grep -v '^{' | tail -6
[ -z "${DRY_RUN:-}" ] && echo "done: $OUT"
