# DeepSeek V4.1 Flash, fast, on one Mac Studio

> **⚠️ macOS 27 (Sept 18):** after upgrading to macOS 27, this branch stalls on
> long prompts -- prefill hangs once the conversation history grows past roughly
> 32k tokens. **On macOS 27, use [antirez/ds4#1073](https://github.com/antirez/ds4/pull/1073)
> (kernelpool) instead, without DSpark.** On the same Mac and the same Q2 file it
> measured 37 tok/s plain decode (vs 30.5 here) and read a real ~80k-token
> project without stalling. DSpark on #1073 currently stalls in agent use on
> macOS 27, so leave it off. Details in [V41-FAST-SETUP.md](V41-FAST-SETUP.md).

This branch is **ivanfioravanti's `ds41f-dspark`** with one change on top:
the parallel Engram reader is ungated for ordinary decode. Everything
that makes this fast except that one commit is his work, or antirez's.

Measured on a Mac Studio M3 Ultra (80 GPU cores, 256 GB, macOS 26.6.2),
`DeepSeek-V4.1-Flash-Q2.gguf`, resident, temperature 0:

| build | short / ~10k ctx | ~115k ctx |
| --- | ---: | ---: |
| antirez/ds4 main | 19.2 | — |
| + ivanfioravanti's optimizations | 26.8 | 24.1 |
| + this branch's Engram ungating | 30.8 | 27.6 |
| + DSpark speculative decoding | **42** | 28.3 |

A live 25k-token coding session runs about **39.5 tok/s**.

## What you need

- Apple Silicon Mac with enough RAM to hold the model resident. The Q2
  GGUF is ~341 GB on disk but only ~152 GiB of main weights stay
  resident; the Engram tables are read from the file every token, so
  **the GGUF must live on a fast internal SSD**.
- 256 GB unified memory for the numbers above. Less means
  `--ssd-streaming`, which drops you to ~15 tok/s -- a different
  configuration entirely.
- Xcode command line tools.

## Build

```sh
git clone -b dspark-engram https://github.com/jjakemaness/ds4.git
cd ds4 && make
```

## Run

```sh
./ds4-agent -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --vision gguf/DeepSeek-V4.1-Flash-Vision.gguf \
  --power 100 --ctx 262144
```

`--power 100` is required; V4.1 refuses throttled operation. Deliberately
no `--ssd-streaming`.

Two environment settings keep the model reclaimable instead of wired, so
it releases RAM when idle at the cost of ~1.2 s on the first request:

```sh
export DS4_METAL_NO_RESIDENCY=1
export DS4_METAL_DISABLE_QUEUE_KEEPALIVE=1
```

`DS4_ENGRAM_SERIAL_DECODE=1` restores the old serial Engram path if you
want to A/B the change on your own hardware.

## DSpark (the 30.8 -> 42 step)

Needs a separate ~8.5 GB support file built from the same checkout
revision as your target GGUF:

```sh
revision=df42c109f1defefcbfcedbe7d905718a12266e40
hf download deepseek-ai/DeepSeek-V4.1-Flash --revision "$revision" \
  --local-dir /tmp/ds41-draft-source \
  --include config.json inference/config.json model.safetensors.index.json \
  model-00044-of-00048.safetensors model-00045-of-00048.safetensors \
  model-00046-of-00048.safetensors
make -C gguf-tools
uv run --with numpy python gguf-tools/deepseek41_dspark.py \
  --hf /tmp/ds41-draft-source --source-revision "$revision" \
  --out gguf/DeepSeek-V4.1-Flash-DSpark-support.gguf
```

Then add `--dspark --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-support.gguf`.

See `docs/SPECULATIVE_DECODING.md` for the full description.

## Honest caveats

- **42 is short-to-mid context on code.** At ~115k tokens it is 28.
- **DSpark's benefit is workload-dependent**, from +41% to about -1% in
  our measurements, and it does not track context size -- it tracks how
  predictable the generated tokens are. Code speculates well; prose and
  dense reasoning do not. ivanfioravanti's published figures of ~6% were
  measured on prose, and they are correct for prose.
- **DSpark output is not byte-identical** -- batched verification changes
  floating-point reduction order. Same weights, same model; use
  `--dspark-strict` to disable speculative acceptance for comparisons.
  The Engram change on its own *is* byte-identical.
- DSpark falls back to ordinary decode on image requests.
- Only tested on M3 Ultra.

## Credit

- **[antirez/ds4](https://github.com/antirez/ds4)** -- DwarfStar itself,
  and the concurrent reader this change reuses.
- **[ivanfioravanti/ds4](https://github.com/ivanfioravanti/ds4)** --
  `ds41f-optimizations` and `ds41f-dspark`; almost everything above the
  19.2 baseline is his, including `ds41_engram_parallel()` itself. This
  branch only changes when it runs.
- **The oMLX / drowzeys V4.1-on-Mac work** -- what pointed at Engram as a
  decode bottleneck in the first place. Their approach was a RAM cache of
  hot rows; measured row reuse here was only 22.5% over a 600-token
  coding generation, so the win turned out to be concurrency rather than
  caching. Wrong idea, right place to look.

The Engram change alone is submitted upstream as
[antirez/ds4#1072](https://github.com/antirez/ds4/pull/1072).
