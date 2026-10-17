"""Preview or upload original downloaded media to a private Hugging Face dataset."""

import argparse
from io import BytesIO
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi

from video_tools import ATTACHMENTS, VIDEO_EXTENSIONS


def source_media(folder: Path = ATTACHMENTS) -> dict[str, Path]:
    """Select only original downloader outputs; exclude answers and transcripts."""
    files = {}
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if path.is_file() and re.fullmatch(r"[A-Za-z0-9_-]{11}", path.stem) and path.suffix.lower() in VIDEO_EXTENSIONS:
                files[f"videos/{path.stem}/video{path.suffix.lower()}"] = path
    downloaded = folder / "videos"
    if downloaded.is_dir():
        for directory in sorted(downloaded.iterdir()):
            if not directory.is_dir() or not re.fullmatch(r"[A-Za-z0-9_-]{11}", directory.name):
                continue
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.stem in {"audio", "video"} and path.suffix.lower() in VIDEO_EXTENSIONS | {".m4a", ".mp3", ".ogg"}:
                    files[f"videos/{directory.name}/{path.name}"] = path
    return {name: path for name, path in files.items() if path.stat().st_size}


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Private dataset ID, e.g. username/agent-video-cache")
    parser.add_argument("--upload", action="store_true", help="Create/verify the private repository and upload the previewed raw files")
    parser.add_argument("--space", help="Also configure this Space's VIDEO_MEDIA_REPO variable and HF_TOKEN secret")
    args = parser.parse_args()
    files = source_media()
    if not files:
        raise SystemExit("No original downloaded video/audio files were found.")
    for destination, path in files.items():
        print(f"{destination}: {path.stat().st_size:,} bytes")
    if not args.upload:
        print("Preview only. Add --upload to publish these source files to the private dataset.")
        return
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN in .env before uploading.")
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(args.repo, repo_type="dataset").private:
        raise SystemExit("Refusing to upload: the target dataset is public. Choose a private repository.")
    readme = (
        "---\nlicense: other\n---\n\n"
        "# Private source media cache\n\n"
        "Original source clips/audio downloaded for this account's study agent. "
        "Use these files for analysis rather than redownloading from hosted servers. "
        "No benchmark answers, transcripts, or model-generated observations are included. "
        "This cache grants no redistribution rights to the underlying source media.\n"
    )
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in files.items()]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=BytesIO(readme.encode())))
    result = api.create_commit(repo_id=args.repo, repo_type="dataset", operations=operations, commit_message="Cache original video and audio inputs for study agent")
    print(f"Uploaded {len(files)} original media files to private dataset {args.repo}.")
    print("Commit:", result.oid)
    if args.space:
        api.add_space_secret(repo_id=args.space, key="HF_TOKEN", value=token)
        api.add_space_variable(repo_id=args.space, key="VIDEO_MEDIA_REPO", value=args.repo)
        print(f"Configured private media access for Space {args.space}.")


if __name__ == "__main__":
    main()
