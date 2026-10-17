"""Inspect a video independently of Gradio and the assignment submission API."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from video_tools import analyze_video, transcribe_video


def main():
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="YouTube URL or path to a video in attachments/")
    parser.add_argument("--question", help="Visual question to investigate")
    parser.add_argument("--audio", action="store_true", help="Transcribe spoken audio instead of inspecting frames")
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--output", type=Path, default=base_dir / "attachments" / "video_report.json")
    args = parser.parse_args()
    params = {"source": args.source, "start_seconds": args.start, "end_seconds": args.end}
    if args.audio:
        result = transcribe_video.invoke(params)
    else:
        if not args.question:
            parser.error("--question is required for visual analysis")
        result = analyze_video.invoke({**params, "question": args.question, "interval_seconds": args.interval})
    report = json.loads(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report: {args.output}")
    if "error" in report:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
