# DeepSeek V4.1 Flash at ~50 tok/s on one Mac Studio — macOS 27 setup

A tested recipe for running DeepSeek V4.1 Flash Q2 fast and without freezes on a
single Mac Studio M3 Ultra (256 GB) on macOS 27.

**The engine is not mine.** This branch is
[antirez/ds4](https://github.com/antirez/ds4) (DwarfStar) with
[kernelpool's PR #1073](https://github.com/antirez/ds4/pull/1073), *pinned at the
exact commit I tested* (`fd70919`). DSpark speculative decoding is
[ivanfioravanti](https://github.com/ivanfioravanti)'s work. What this branch adds
is the recipe: the macOS 27 fix, a way to build the DSpark file without
downloading the whole model, launch scripts, and measurements.

## Results

Mac Studio M3 Ultra, 80-core GPU, 256 GB, macOS 27.0, `DeepSeek-V4.1-Flash-Q2.gguf`,
DSpark on, default memory settings:

| conversation size | writing | reading |
|---|---:|---:|
| short answers | ~46–50 tok/s (41 on a code-writing benchmark) | — |
| ~10k tokens | 47 tok/s | 508 tok/s |
| ~115k tokens | 34 tok/s | ~400–500 tok/s |
| follow-up in a 115k chat (server) | answered in **2 s** — reused 114,810 of 114,826 tokens | |

Without DSpark (`NOSPEC=1`): ~38 tok/s, byte-identical output. For reference,
stock DwarfStar main measured 19.2 tok/s on the same machine.

## The macOS 27 freeze, and the fix

After updating to macOS 27, long prompts froze for minutes: GPU at 0%, the program
stuck inside Apple's GPU driver (`-[IOGPUMetalCommandQueue submitCommandBuffers:count:]`).

The cause was two settings that are popular for releasing RAM when the model is idle:

```sh
DS4_METAL_NO_RESIDENCY=1              # DON'T
DS4_METAL_DISABLE_QUEUE_KEEPALIVE=1   # DON'T
```

With them set, macOS drops its GPU map of the ~155 GB model whenever the engine idles
for a few seconds, and on macOS 27 rebuilding that map can hang for minutes. **Leave
both unset.** The model then stays pinned in RAM (~160 GB) while it runs and is freed
when it exits. The session that froze (five-file read, one-minute idle, then an
8,437-line file read, up to 219.7k context) completed twice after the change.

Two related things worth knowing: don't run a second large model at the same time,
and conversations get slower as they grow (use a fresh session between tasks).

## Setup

```sh
git clone -b v41-macos27 https://github.com/jjakemaness/ds4.git
cd ds4
make ds4 ds4-agent ds4-server

./download_model.sh ds41f-q2        # ~341 GB; keep it on the internal SSD
./download_model.sh ds41f-vision

scripts/build-dspark-support.sh     # ~8 GB download instead of ~330 GB
```

`build-dspark-support.sh` is the trick: the converter plans the whole model even
when only writing the DSpark file, so it expects all 48 checkpoint shards — but it
only *reads* shards 44–46. The script downloads those three and replaces the other
45 with header-only placeholders (~10 MB on disk). Add `DRY_RUN=1` to check first.

## Run

```sh
scripts/serve-v41.sh                          # server on http://127.0.0.1:8002/v1
scripts/chat-v41.sh                           # server + chat client in one terminal
scripts/agent-v41.sh --chdir ~/your/project   # DwarfStar's own coding agent
```

`chat-v41.sh` uses `CHAT_CMD` (default `hermes -p v41 chat`); point any
OpenAI-compatible client at port 8002. Only one of these can run at a time.

Options for all three: `NOSPEC=1` (no DSpark), `CTX=524288` (bigger context
window), `MODEL_DIR=/path/to/gguf`.

## Caveats

- DSpark's gain depends on the text: large on code and short answers, small on
  prose. Output with DSpark is not byte-identical (batched verification changes
  floating-point order); `NOSPEC=1` if you need that.
- Image turns fall back to ordinary decoding.
- Tested on one M3 Ultra. Other machines will differ.

## Credit

- [antirez/ds4](https://github.com/antirez/ds4) — DwarfStar
- [kernelpool, PR #1073](https://github.com/antirez/ds4/pull/1073) — the Metal speedups this runs on
- [ivanfioravanti](https://github.com/ivanfioravanti/ds4) — DSpark and earlier V4.1 optimizations
- The oMLX / drowzeys V4.1-on-Mac work, which pointed at the Engram bottleneck early on

Findings posted upstream: [PR #1073 results](https://github.com/antirez/ds4/pull/1073#issuecomment-5736684935),
[issue #931](https://github.com/antirez/ds4/issues/931#issuecomment-5736690073).
