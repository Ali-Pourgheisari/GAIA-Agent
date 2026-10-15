"""Configure local Hugging Face Git authentication using the project's .env."""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN in your .env file before running this script.")

    login(token=token, add_to_git_credential=True)


if __name__ == "__main__":
    main()
