"""Extract timestamped frames without making any model calls."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from video_tools import extract_frames, sample_times, video_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=Path(__file__).resolve().parent / "attachments" / "L1vXCYZAYYM.mp4")
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    args = parser.parse_args()
    video_path = args.video.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    info = video_info(video_path)
    end = (info.frame_count - 1) / info.fps
    if args.end is not None:
        end = min(end, args.end)
    frames = extract_frames(video_path, sample_times(args.start, end, args.interval))
    manifest = frames[0].path.parent / "frames.json"
    manifest.write_text(json.dumps({
        "video": str(video_path),
        **asdict(info),
        "frames": [{"timestamp_seconds": frame.timestamp_seconds, "path": str(frame.path)} for frame in frames],
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(frames)} frames. Timestamp manifest: {manifest}")


if __name__ == "__main__":
    main()
