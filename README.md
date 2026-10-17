---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Video tools

Install the project requirements in your virtual environment. Set `OPENAI_API_KEY`
in the local `.env` or in your Hugging Face Space's Secrets. No key is stored in
the source code. `VIDEO_MODEL` optionally selects the vision model (default:
`gpt-4o`). Video analysis and transcription make paid API calls.

The LangGraph agent can call two video tools:

- `analyze_video(source, question, start_seconds=0, end_seconds=None,
  interval_seconds=2)` downloads a short YouTube clip or opens a local video
  inside `attachments/`. It returns visual observations with timestamps, counts
  per frame, and uncertainty. It samples the requested range and refines around
  two high-count moments. It never adds counts from different frames together.
- `transcribe_video(source, start_seconds=0, end_seconds=None)` downloads the
  audio stream when given a YouTube URL, extracts mono audio using the bundled
  FFmpeg executable, and returns speech plus approximate segment timestamps.
  Use it for spoken dialogue, rather than visual species counts.

MP4, WebM, MOV, MKV and AVI attachments are passed to the graph with their local
path so these tools can inspect them. Downloads, extracted frames, transcripts,
and analysis caches stay in `attachments/`, which Git ignores. Successful
analysis/transcription is cached using the source file and analysis settings.
The final formatter still prepares the answer for assignment submission.

### Local checks (no submission)

Extract frames and a timestamp manifest without calling a model:

```bash
python extract_frames.py attachments/L1vXCYZAYYM.mp4 --interval 2
```

Inspect visual evidence (uses the model):

```bash
python video_cli.py 'https://www.youtube.com/watch?v=L1vXCYZAYYM' \
  --question 'What is the highest number of bird species visible simultaneously?'
```

Investigate a smaller interval more densely:

```bash
python video_cli.py 'https://www.youtube.com/watch?v=L1vXCYZAYYM' \
  --question 'How many distinct bird species are visible together?' \
  --start 25 --end 40 --interval 0.5
```

Transcribe video dialogue:

```bash
python video_cli.py 'https://www.youtube.com/watch?v=1htKBjuUWec' --audio
```

Run offline checks with model/transcription responses mocked:

```bash
python -m unittest discover -s tests -v
```

### Coverage and errors

Visual analysis is based on sampled frames. Its `observed_maximum` is a model
estimate over those frames, not proof of a maximum across every instant. The
report includes sample spacing and uncertainty; brief appearances and species
identification can still be missed. Default sampling uses at most 96 frames,
plus up to 32 refinement frames. For longer clips the spacing increases to cover
the requested interval. Downloaded clips are limited to 15 minutes/150 MB.
Video timestamps use the decoder's reported frame rate and are approximate for
variable-frame-rate videos. Speech transcription can also contain errors.

YouTube availability and download restrictions vary between a local machine and
hosted Spaces. Download/API errors are returned to the agent as errors; they are
not treated as answers. No browser cookies or account passwords are required by
this implementation. The CLI and extraction commands do not submit benchmark
answers or publish code.
