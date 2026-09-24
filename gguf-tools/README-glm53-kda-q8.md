# glm53_kda_q8_inplace.py

Rewrites a `GLM-5.3-Flash-Q4_K.gguf` in place so its KDA attention weights and
output head are `Q8_0` instead of `BF16`.

## Why

`glm53_quantize.py`'s q4 recipe already asks for this:

```python
if role in ("embedding", "output"):
    return QTYPE_Q8_0
if role == "linear_attention":
    ...
    return QTYPE_Q8_0
```

The GGUF published at `antirez/glm-5.3-flash-gguf` (2026-08-28) predates that
recipe commit (2026-09-17) and ships the `linear_attention` group as `BF16`.

Two consequences on a fully-resident setup:

- `kda_q/k/v` plus the output head are **7.5 GiB of the 17.8 GiB read per
  decoded token**. At `BF16` that is about 54% of decode bandwidth.
- `ds4_gpu_glm53_kda_inputs_q8_bf16()` requires `kda_q/k/v` to be `Q8_0`, so on
  the published file that fast path never engages.

Measured on a 256 GB M3 Ultra, ds4-server, 400-token generations, median of 3:
**28.4 -> 34.8 tok/s decode**, prefill unchanged at ~513 tok/s.

Rebuilding from the official FP8 snapshot is the cleaner fix, but needs
hundreds of GiB. This rewrites the file you already have.

## Usage

```
python3 glm53_kda_q8_inplace.py MODEL.gguf --dry-run   # print the plan
python3 glm53_kda_q8_inplace.py MODEL.gguf             # rewrite in place
```

Then serve with `DS4_GLM_ENABLE_KDA_Q8_INPUTS=1`.

Requires numpy. Idempotent: on an already-converted file it reports zero
tensors and exits.

## What it converts, and what it must not

| Tensor | Action | Reason |
| --- | --- | --- |
| `blk.*.kda_q` / `kda_k` / `kda_v` | -> `Q8_0` | the bulk of the bytes; enables the fused Q8 input path |
| `output.weight` | -> `Q8_0` | recipe specifies it; no Metal path requires `BF16` |
| `blk.*.kda_output` | left `BF16` | the HC-expand epilogue checks for `BF16` |
| `kda_f_a` / `f_b` / `g_a` / `g_b` / `beta` | left `BF16` | the KDA gate trio checks for `BF16` |
| `token_embd` | left `BF16` | row lookup, no per-token bandwidth benefit |

Converting the last three makes ds4 fall back to slower paths, costing more
than the saved bytes.

## How in-place is safe

Every converted tensor shrinks (`BF16` 2 bytes/weight -> `Q8_0` 1.0625) and
nothing else moves, so walking tensors in file order the write cursor always
trails the read cursor. The header keeps its byte length because the tensor
type field is a fixed-width `uint32`. The file is truncated at the end. Peak
disk usage is what the file already occupies.

The script asserts `min(read_offset - write_offset) >= 0` across the whole plan
before writing, and refuses otherwise.

## Correctness

The `Q8_0` encoder is byte-identical to `ds4q_quantize_q8_0()` in `quants.c`,
verified on random vectors. Two details matter if you reimplement it: `id` is
computed from the `f32` scale rather than the `f16`-rounded one, and C's
`roundf()` rounds half away from zero where numpy's default is banker's
rounding.

After converting, check: tensor count unchanged, no overlapping offsets, the
last tensor ending exactly at EOF, and the `Q8_0` count up by the number
reported.

## Caveats

Destructive and non-resumable. A crash mid-write leaves an unusable file and
you re-download. There is no undo and no backup is made.

Tested on one file, on one machine (256 GB M3 Ultra, macOS 27). Quality
spot-checks after conversion (arithmetic, factual recall, code generation) were
clean, but this has not been run through a perplexity or benchmark suite.
