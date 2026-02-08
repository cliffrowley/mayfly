# Mayfly - AI‑Assisted Archive Relearning (MVP)

## Purpose

Create the **smallest thing that could possibly work** to help us re‑learn old tracks we’ve forgotten how to play.

This is a deliberately minimal, disposable tool:

* a **command‑line utility**
* run manually, on demand
* produces **Markdown files** as memory prompts

The tool exists to jog memory, not to be a long‑lived system.

## Core Use Case

> “I point a tool at a directory of old audio files and quickly get enough information back to remember how the song goes.”

The output should:

* reduce blank‑page paralysis
* remind us of key, tempo, and lyrical cues
* tell us *where to listen* if something is uncertain

Once a track is re‑learned, the output may never be used again.

## Constraints & Assumptions

* Recordings span decades
* Mixed quality (studio, demo, rehearsal, live)
* **No stems available**
* Tonal, song‑based music
* Imperfect results are acceptable

## MVP Shape (KISS / YAGNI)

### Execution Model

* A **command‑line program**
* Accepts a single audio file or recursively scans a directory
* Processes each file independently
* Supports selective re‑runs via `--only` (e.g. `--only lyrics`)
* Emits **one `.txt` file (Markdown‑formatted) per audio file**

No database. No UI. No service. No persistence beyond files on disk.

## Minimal Analysis Pipeline

Each step is best‑effort. Failure of any step must not break the tool.

### 1. Discovery

* Recursively find audio files
* Supported formats: wav, mp3, aiff (others best‑effort)

### 2. Starting Key Detection

* Analyse only the first 15–30 seconds
* Output:

  * starting key
  * confidence
  * optional alternate (e.g. relative major/minor)

### 3. Tempo Detection

* Estimate BPM
* Output confidence

### 4. Lyrics Draft

* Best‑effort transcription from full mix
* Time‑stamped lines
* Explicit markers for low‑confidence or unintelligible spans

## Libraries & Models (MVP Choices)

The MVP prioritises **minimum effort and maximum leverage**. All choices are made to reduce glue code and setup friction.

### Language

* **Python**
* Single‑script mindset (not a framework or package)

### Audio I/O & Musical Analysis

* `librosa`

  * Audio loading
  * Tempo (BPM) estimation
  * Chroma features for starting‑key detection
* `numpy`
* `soundfile`

Rationale:

* Mature and well‑understood
* One import away from useful musical features
* Entirely sufficient for *starting key only* analysis

### Lyrics Transcription

* **`faster‑whisper`**

  * Singing‑tolerant speech‑to‑text
  * Time‑stamped segments
  * Best‑effort transcription on full mixes

Model guidance:

* Prefer `small` or `medium`
* `large` is unnecessary for MVP

### Output Generation

* `pyyaml`

  * YAML frontmatter generation
* Plain string‑based Markdown output

  * No templating engine
  * No schema enforcement

### Explicitly Deferred / Excluded

The following are intentionally **not** part of the MVP:

* Source separation (e.g. Demucs)
* Chord‑detection models
* Databases or persistence layers
* Config files or plugin systems
* Training or fine‑tuning models
* GPU‑only dependencies

## Output Format

### One Markdown File per Track

Markdown is the only output.

The file should be:

* human‑readable
* easy to skim
* easy to edit or delete

### Frontmatter (MVP Only)

YAML frontmatter containing only:

* source_file
* starting_key (+ confidence, optional alternate)
* tempo_bpm (+ confidence)

### Markdown Body

* Draft lyrics with timestamps
* Low‑confidence spans clearly marked
* Optional free‑form notes added manually if desired

## Confidence & Uncertainty

Confidence is a core feature.

The tool must clearly indicate:

* when an output is uncertain
* *where* in the track uncertainty occurs (timestamps)

Confidence exists solely to direct human listening effort.

The tool should prefer:

* clearly flagging uncertainty
* over pretending accuracy

## Non‑Goals (Explicit)

This MVP does **not** aim to:

* be interactive
* support play‑along or rehearsal modes
* provide perfect transcription
* persist state or corrections
* scale beyond local, manual use

## Success Criteria

The MVP is successful if:

* running it on a directory of forgotten tracks
* produces Markdown files that
* allow us to remember how to play a song in minutes rather than hours

Nothing more is required.
