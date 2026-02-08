# Mayfly

A quick-and-dirty CLI that scans a directory of old audio recordings and generates memory-prompt files — starting key, tempo, and draft lyrics — so you can re-learn forgotten songs in minutes rather than hours.

## Install

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```sh
# Scan a whole directory
python main.py /path/to/audio

# Process a single file
python main.py /path/to/song.mp3

# Re-run only specific steps (key, tempo, lyrics)
python main.py /path/to/song.mp3 --only lyrics
python main.py /path/to/song.mp3 --only key,tempo

# Force regeneration of existing output
python main.py /path/to/audio --force

# Write output to a different directory
python main.py /path/to/audio -o /path/to/output
```

By default, files that already have output are skipped. Use `--force` to regenerate them. Using `--only` always re-processes the selected steps, preserving the rest from the existing output.

## What It Produces

One `.txt` file per audio file, containing:

- **YAML frontmatter** — source file path, starting key (with confidence), tempo BPM (with confidence)
- **Markdown body** — key/tempo summary and timestamped draft lyrics

Low-confidence results are flagged with ⚠️ so you know where to listen more carefully.

### Example Output

```
---
source_file: demos/Old Song.mp3
starting_key: A minor
starting_key_confidence: medium
tempo_bpm: 120.5
tempo_confidence: medium
starting_key_alternate: C major
---

# Old Song

**Key:** A minor
**Possible alternate:** C major
**Tempo:** 120.5 BPM

## Draft Lyrics

[0:00] Here are some words that were transcribed
[0:08] And this line was less certain  ⚠️
```

## Supported Formats

wav, mp3, aiff — others are attempted best-effort.

## Limitations

- Transcription is from the **full mix** (no source separation) — expect imperfect lyrics, especially from dense mixes
- Key detection uses only the **first ~30 seconds**, so key changes mid-song won’t be caught
- Confidence indicators are heuristic, not ground truth
- This is a disposable tool, not a production system
