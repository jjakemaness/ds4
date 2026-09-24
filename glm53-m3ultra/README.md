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

The two gains compound rather than add: +29% from the PR, then +22% on top of
that, for +58% overall.

For reference, ds4's QA table lists 24.74 tok/s decode and 437.62 t/s prefill
for this model on a **512 GB** M3 Ultra. Prefill on 256 GB matched to within
0.1%.

## Before you start

- **256 GB Apple Silicon.** The model is 178 GiB resident. A 512 GB machine
  should be at least as fast; 128 GB will not hold it.
- **~180 GiB free disk.** The conversion is in place and needs no extra space,
  but the download does.
- **Check your GPU memory limit.** macOS caps how much unified memory the GPU
  may wire, and a low cap makes a 178 GiB model fail to load with an unhelpful
  error:
  ```
  sysctl iogpu.wired_limit_mb
  ```
  `0` means no explicit cap, which is fine. A small number (a few thousand) is
  not — some machines carry one over from a migrated LaunchDaemon. Raise it:
  ```
  sudo sysctl -w iogpu.wired_limit_mb=250000
  ```
  Add it to a LaunchDaemon if you want it to survive reboots. Leave the OS at
  least ~8 GB.
- **Python 3 with numpy**, for the converter: `pip install numpy`.
- Measured on **macOS 27** (Darwin 27.0.0). Other versions untested.

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
   The path matters: `serve-glm-q4.sh` defaults to `~/ds4-m3perf`, and ds4 loads
   its Metal shaders relative to the build directory. Override with `BUILD=`.

3. **Fix the checkpoint.** The published GGUF ships its KDA attention weights at
   BF16; ds4's own quantizer recipe says Q8_0. The converter lives at the repo
   root, in `gguf-tools/`:
   ```
   python3 gguf-tools/glm53_kda_q8_inplace.py ~/ds4/gguf/GLM-5.3-Flash-Q4_K.gguf --dry-run
   python3 gguf-tools/glm53_kda_q8_inplace.py ~/ds4/gguf/GLM-5.3-Flash-Q4_K.gguf
   ```
   Run the dry run first and read what it plans. The real run is destructive and
   non-resumable: if it dies mid-write the file is unusable and you re-download.
   Takes about 3 minutes for 178 GiB. It is idempotent, so a second run on an
   already-converted file reports zero tensors and exits.

   Details and the full tensor table: `gguf-tools/README-glm53-kda-q8.md`.

4. **Serve:**
   ```
   ./glm53-m3ultra/serve-glm-q4.sh
   ```
   Override paths with `BUILD=`, `G=` (gguf dir), `CTX=`, `PORT=`.

`DS4_GLM_ENABLE_KDA_Q8_INPUTS=1` is set by the script and is required. It gates
the Q8 KDA input path, which only works once the weights are actually Q8_0 —
setting it before the conversion does nothing.

## Checking you got it

```
curl -s localhost:8003/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm","messages":[{"role":"user","content":"Explain consensus algorithms in distributed systems."}],"max_tokens":400}' >/dev/null
```
Then read the decode rate off the server log. Expect ~34-35 tok/s.

`sweep-glm.sh` and `confirm-glm.sh` automate this across configs, reloading the
model each time. Use 400-token generations and medians of three: single-run
spread here is about 1.5 tok/s, which is enough to invent a trend that is not
there. I nearly published one.

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
~76 ms against a ~45 ms plain decode step, so verification costs more than it
saves even when drafts are accepted.

## Why it was slow

Decode on a fully-resident model is bound by bytes read per token:

| Tensor group | Bytes/token | Share |
| --- | ---: | ---: |
| KDA attention (34 layers, BF16) | 8.48 GiB | 48% |
| Routed experts (8 of 288 active) | 4.54 GiB | 26% |
| DSA attention (12 layers, Q8_0) | 1.37 GiB | 8% |
| Head, shared expert, dense FFN | 3.38 GiB | 19% |
| **Total** | **17.77 GiB** | |

GLM-5.3-Flash interleaves two attention types — DSA every fourth layer, KDA in
the other 34. The DSA layers ship at Q8_0; the KDA layers ship at BF16. Half the
per-token bandwidth was going to tensors stored at twice the precision of their
neighbours. Converting takes it to 14.24 GiB/token.

A CPU profile pointed at command-buffer submission and was misleading — that
work runs on a separate dispatch thread, concurrent with the GPU. Counting bytes
found the real problem.

## Credit

- [antirez](https://github.com/antirez/ds4) — DwarfStar (ds4), and the
  quantisation recipe that made the stale checkpoint findable.
- [trueimage](https://github.com/antirez/ds4/issues/1090) — PR #1090, the
  larger of the two speedups.
- Zhipu / the GLM team — the model.

Mine: spotting that the published Q4_K predates the recipe, the in-place
converter, the 256 GB reproduction, and the negative results.
