---
title: GAIA Agent
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

# GAIA Agent

A LangGraph agent for the Hugging Face Agents Course GAIA assignment, with web
research, attachment handling, video analysis, and audio transcription tools.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Missing course attachments

The downloader reuses nonempty files in `attachments/`, then tries the course
API. If the course API returns 404, it requests only the named attachment from
`2023/validation/` in the official `gaia-benchmark/GAIA` dataset. No benchmark
answers or metadata are downloaded by this fallback.

The official dataset is gated: accept its conditions while logged into your
Hugging Face account at https://huggingface.co/datasets/gaia-benchmark/GAIA.
Your `HF_TOKEN` must belong to an account with access and permit reading the
gated dataset. Set it in local `.env` or as a Space **Secret**. Alternatively,
place the original attachment under its expected filename in `attachments/`.
Keep downloaded dataset files private; the attachments directory is ignored by Git.

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

### Hosted YouTube downloads blocked

When YouTube rejects a Space's IP with a bot-confirmation error, use a private
cache of original media. Download on your local machine first, then preview:

```bash
python prepare_video_cache.py --repo YOUR_USERNAME/gaia-agent-media
```

The script selects only original video/audio downloads, excluding `.env`,
transcripts, answer files, frame observations, and benchmark metadata. To upload
and configure your Space explicitly:

```bash
python prepare_video_cache.py --repo YOUR_USERNAME/gaia-agent-media \
  --upload --space YOUR_USERNAME/GAIA-Agent
```

The target dataset must be private. The script sets the Space's `HF_TOKEN`
secret and `VIDEO_MEDIA_REPO` variable. These video tools look in local storage,
then that private dataset, then attempt YouTube. The Space still runs the actual
vision/transcription analysis on the source files; cached answers are not used.
Push the code changes to the Space after configuring it.

The existing deployment remains at
https://huggingface.co/spaces/gheysar4real/Final_Assignment_Template; its repository
ID is retained so existing deployment links and configuration continue to work.
