# GLM-5.3-Flash Q4 at ~35 tok/s on a 256 GB M3 Ultra

Exact config behind these numbers. Nothing here is a tuning flag.

| Stage | Decode | Prefill @2048 | Wired |
| --- | ---: | ---: | ---: |
| Published GGUF + ds4 main | 22.0 tok/s | 437 t/s | 185 GiB |
| + PR #1090 | 28.4 tok/s | 515 t/s | 174 GiB |
| + Q8_0 KDA conversion | **34.8 tok/s** | 513 t/s | 175 GiB |

Decode is ds4-server's own counter. The final figure is a median of three
400-token generations (34.86 / 34.80 / 34.81). The first two rows were measured
with 128-token generations, which read ~10% low; measured that way the same
three stages are 22.0 -> 28.4 -> 31.3.

For reference, ds4's QA table lists 24.74 tok/s decode and 437.62 t/s prefill
for this model on a **512 GB** M3 Ultra. Prefill on 256 GB matched to within
0.1%.

## Setup

1. **Model** (178 GiB):
   ```
   hf download antirez/glm-5.3-flash-gguf GLM-5.3-Flash-Q4_K.gguf --local-dir ~/ds4/gguf
   hf download antirez/glm-5.3-flash-gguf GLM-5.3-Flash-Vision-Encoder.gguf --local-dir ~/ds4/gguf
   ```

2. **Engine** — trueimage's PR #1090, unmerged:
   ```
   git clone --branch ds41f-m3ultra-perf https://github.com/trueimage/ds4.git ~/ds4-m3perf
   cd ~/ds4-m3perf && make -j$(sysctl -n hw.ncpu) ds4-server
   ```

3. **Fix the checkpoint.** The published GGUF ships its KDA attention weights at
   BF16; ds4's own quantizer recipe says Q8_0. Convert in place (~3 min, no
   extra disk):
   ```
   python3 gguf-tools/glm53_kda_q8_inplace.py ~/ds4/gguf/GLM-5.3-Flash-Q4_K.gguf --dry-run
   python3 gguf-tools/glm53_kda_q8_inplace.py ~/ds4/gguf/GLM-5.3-Flash-Q4_K.gguf
   ```
   Destructive and non-resumable. See `gguf-tools/README-glm53-kda-q8.md`.

4. **Serve:**
   ```
   ./glm53-m3ultra/serve-glm-q4.sh
   ```

`DS4_GLM_ENABLE_KDA_Q8_INPUTS=1` is set by the script and is required — it
gates the Q8 KDA input path, which only works once the weights are Q8_0.

## Benchmarking

`sweep-glm.sh` and `confirm-glm.sh` reload the model per config and report
ds4-server's decode counter. Use 400-token generations and medians of three;
single-run spread here is about 1.5 tok/s, which is enough to invent a trend
that is not there.

## Things that did not help

| Attempt | Result |
| --- | --- |
| Native MTP (`--mtp`) | 18.7 tok/s, 15% slower |
| DFlash2 external drafter | `--mtp-model is not supported for GLM yet` |
| `DS4_GLM_DECODE_FLUSH_INTERVAL=0` | 20.4 tok/s, slower |
| `DS4_GLM_ENABLE_TOPK_FAST=1` | no effect; gate needs `top_k == 512`, model ships 2048 |
| `DS4_METAL_Q8_MV_NSG` 2 or 8 | both worse; default 4 is right |
| `DS4_GLM_DECODE_SPLIT_BLOCK_ROWS` 4-64 | all within 0.14 tok/s of default |

Speculative decoding loses on this chip. `--mtp-timing` puts the verify pass at
~76 ms against a ~45 ms plain decode step.

## Credit

- [antirez](https://github.com/antirez/ds4) — DwarfStar (ds4), and the
  quantisation recipe that made the stale checkpoint findable.
- [trueimage](https://github.com/antirez/ds4/issues/1090) — PR #1090, the
  larger of the two speedups (+29% of the +58%).
- Zhipu / the GLM team — the model.

Mine: spotting that the published Q4_K predates the recipe, the in-place
converter, the 256 GB reproduction, and the negative results.
