# Mayfly — Copilot Instructions

## What This Project Is

Mayfly is a **disposable CLI tool** that scans a directory of old audio recordings and emits one `.txt` file (Markdown-formatted) per track containing starting key, tempo, and draft lyrics. It exists to jog musicians' memories, not to be a long-lived system. See [README.md](../README.md) for the full spec.

## Architecture & Design Philosophy

- **Single-script CLI** — all logic lives in `main.py`. Do not introduce packages, modules, frameworks, or plugin systems.
- **No persistence** — no database, no config files, no state beyond the generated `.txt` files on disk.
- **Best-effort pipeline** — the analysis pipeline has ordered steps (discovery → key → tempo → lyrics). Failure of any individual step must never crash the tool; degrade gracefully and flag uncertainty in the output.
- **Confidence is a first-class concept** — every analysis result should carry a confidence indicator. Prefer clearly flagging uncertainty over pretending accuracy.

## Tech Stack (MVP — do not expand)

- **Python** (single-script mindset)
- `librosa` — audio loading, BPM estimation, chroma features for key detection
- `numpy`, `soundfile` — audio I/O support
- `faster-whisper` (model: `small` or `medium`) — lyrics transcription from full mixes
- `pyyaml` — YAML frontmatter generation
- Plain string concatenation for Markdown body — **no templating engines**

### Explicitly excluded

Do not introduce: source separation (Demucs), chord-detection models, GPU-only deps, config/plugin systems, databases, or training/fine-tuning code.

## Key Conventions

- **Output format**: one `.txt` file per audio file with YAML frontmatter (`source_file`, `starting_key`, `tempo_bpm`) and a Markdown-formatted body (timestamped draft lyrics with low-confidence markers).
- **Key detection**: analyse only the **first 15–30 seconds** of each track.
- **Supported input formats**: wav, mp3, aiff (others best-effort).
- **Discovery**: recursive directory scan for audio files.
- **Error handling**: wrap each pipeline step in its own try/except; log the failure and continue to the next step or the next file.

## Running

```sh
python main.py <directory-of-audio-files>
python main.py <single-audio-file>
python main.py <single-audio-file> --only lyrics
python main.py <single-audio-file> --only key,tempo
```

Use `--only` to selectively re-run pipeline steps (`key`, `tempo`, `lyrics`). Steps not listed are preserved from the existing output file.

By default, files with existing output are skipped. Use `--force` to regenerate them.
