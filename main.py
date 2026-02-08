#!/usr/bin/env python
"""Mayfly — disposable CLI for re-learning forgotten songs from audio files."""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger("mayfly")

# ── Constants ────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {".wav", ".mp3", ".aiff", ".aif", ".flac", ".ogg", ".m4a"}
KEY_ANALYSIS_DURATION = 30.0  # seconds of audio to analyse for key
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
WHISPER_MODEL = "small"


# ── Discovery ────────────────────────────────────────────────────────────────

def discover_audio_files(directory: Path) -> list[Path]:
    """Recursively find audio files in *directory*."""
    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    log.info("Found %d audio file(s) in %s", len(files), directory)
    return files


# ── Key Detection ────────────────────────────────────────────────────────────

def detect_key(path: Path) -> dict:
    """Estimate the starting key from the first ~30 s of audio.

    Returns a dict with ``key``, ``confidence``, and ``alternate``.
    """
    import librosa
    import numpy as np

    try:
        y, sr = librosa.load(str(path), duration=KEY_ANALYSIS_DURATION, sr=None)
    except Exception as exc:
        log.warning("Could not load audio for key detection (%s): %s", path.name, exc)
        return {"key": "unknown", "confidence": "none", "alternate": None}

    try:
        chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_energy = np.mean(chromagram, axis=1)

        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        major_corrs = np.array([
            np.corrcoef(np.roll(major_profile, i), chroma_energy)[0, 1]
            for i in range(12)
        ])
        minor_corrs = np.array([
            np.corrcoef(np.roll(minor_profile, i), chroma_energy)[0, 1]
            for i in range(12)
        ])

        best_major_idx = int(np.argmax(major_corrs))
        best_minor_idx = int(np.argmax(minor_corrs))
        best_major_corr = float(major_corrs[best_major_idx])
        best_minor_corr = float(minor_corrs[best_minor_idx])

        if best_major_corr >= best_minor_corr:
            key = f"{KEY_NAMES[best_major_idx]} major"
            confidence = _corr_to_confidence(best_major_corr)
            alt_key = f"{KEY_NAMES[best_minor_idx]} minor"
        else:
            key = f"{KEY_NAMES[best_minor_idx]} minor"
            confidence = _corr_to_confidence(best_minor_corr)
            alt_key = f"{KEY_NAMES[best_major_idx]} major"

        return {"key": key, "confidence": confidence, "alternate": alt_key}

    except Exception as exc:
        log.warning("Key detection failed (%s): %s", path.name, exc)
        return {"key": "unknown", "confidence": "none", "alternate": None}


def _corr_to_confidence(corr: float) -> str:
    if corr > 0.8:
        return "high"
    if corr > 0.5:
        return "medium"
    return "low"


# ── Tempo Detection ──────────────────────────────────────────────────────────

def detect_tempo(path: Path) -> dict:
    """Estimate BPM from the full track.

    Returns a dict with ``bpm`` and ``confidence``.
    """
    import librosa

    try:
        y, sr = librosa.load(str(path), sr=None)
    except Exception as exc:
        log.warning("Could not load audio for tempo detection (%s): %s", path.name, exc)
        return {"bpm": None, "confidence": "none"}

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
        bpm = round(bpm, 1)

        # Heuristic confidence: librosa returns a single estimate, so we base
        # confidence on whether the value falls in a plausible musical range.
        if 50 < bpm < 200:
            confidence = "medium"
        else:
            confidence = "low"

        return {"bpm": bpm, "confidence": confidence}

    except Exception as exc:
        log.warning("Tempo detection failed (%s): %s", path.name, exc)
        return {"bpm": None, "confidence": "none"}


# ── Lyrics Transcription ─────────────────────────────────────────────────────

def transcribe_lyrics(path: Path) -> list[dict]:
    """Best-effort lyrics transcription via faster-whisper.

    Returns a list of segment dicts with ``start``, ``end``, ``text``,
    and ``confidence`` keys.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.warning("faster-whisper not installed — skipping lyrics for %s", path.name)
        return []

    try:
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        segments_iter, _info = model.transcribe(str(path), beam_size=3)

        segments = []
        for seg in segments_iter:
            conf = "low" if seg.avg_logprob < -1.0 else "medium" if seg.avg_logprob < -0.5 else "high"
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "confidence": conf,
            })
        return segments

    except Exception as exc:
        log.warning("Lyrics transcription failed (%s): %s", path.name, exc)
        return []


# ── Markdown Output ──────────────────────────────────────────────────────────

def _fmt_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def build_markdown(source_file: str, key_info: dict, tempo_info: dict,
                   lyrics: list[dict]) -> str:
    """Build the complete Markdown string for one track."""

    # ── Frontmatter ──
    frontmatter = {
        "source_file": source_file,
        "starting_key": key_info["key"],
        "starting_key_confidence": key_info["confidence"],
        "tempo_bpm": tempo_info["bpm"],
        "tempo_confidence": tempo_info["confidence"],
    }
    if key_info.get("alternate"):
        frontmatter["starting_key_alternate"] = key_info["alternate"]

    parts = ["---"]
    parts.append(yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).rstrip())
    parts.append("---")
    parts.append("")

    # ── Body ──
    parts.append(f"# {Path(source_file).stem}")
    parts.append("")

    # Key & tempo summary
    key_str = key_info["key"]
    if key_info["confidence"] == "low":
        key_str += " ⚠️ low confidence"
    parts.append(f"**Key:** {key_str}")

    if key_info.get("alternate"):
        parts.append(f"**Possible alternate:** {key_info['alternate']}")

    tempo_str = f"{tempo_info['bpm']} BPM" if tempo_info["bpm"] else "unknown"
    if tempo_info["confidence"] == "low":
        tempo_str += " ⚠️ low confidence"
    parts.append(f"**Tempo:** {tempo_str}")
    parts.append("")

    # Lyrics
    if lyrics:
        parts.append("## Draft Lyrics")
        parts.append("")
        for seg in lyrics:
            ts = _fmt_timestamp(seg["start"])
            line = f"[{ts}] {seg['text']}"
            if seg["confidence"] == "low":
                line += "  ⚠️"
            parts.append(line)
        parts.append("")
    else:
        parts.append("## Draft Lyrics")
        parts.append("")
        parts.append("*No lyrics could be transcribed.*")
        parts.append("")

    return "\n".join(parts)


# ── Pipeline Orchestration ───────────────────────────────────────────────────

def process_file(audio_path: Path, output_dir: Path) -> None:
    """Run the full analysis pipeline for a single audio file."""
    log.info("Processing: %s", audio_path.name)

    key_info = detect_key(audio_path)
    tempo_info = detect_tempo(audio_path)
    lyrics = transcribe_lyrics(audio_path)

    md = build_markdown(
        source_file=str(audio_path),
        key_info=key_info,
        tempo_info=tempo_info,
        lyrics=lyrics,
    )

    out_path = output_dir / f"{audio_path.stem}.txt"
    out_path.write_text(md, encoding="utf-8")
    log.info("  → %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mayfly — scan audio files and generate memory-prompt Markdown.",
    )
    parser.add_argument("directory", type=Path, help="Directory of audio files to scan")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory for Markdown files (default: same as input)",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        log.error("Not a directory: %s", args.directory)
        sys.exit(1)

    output_dir = args.output or args.directory
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = discover_audio_files(args.directory)
    if not audio_files:
        log.warning("No audio files found.")
        sys.exit(0)

    for audio_path in audio_files:
        try:
            process_file(audio_path, output_dir)
        except Exception as exc:
            log.error("Failed to process %s: %s", audio_path.name, exc)
            continue

    log.info("Done. Processed %d file(s).", len(audio_files))


if __name__ == "__main__":
    main()
