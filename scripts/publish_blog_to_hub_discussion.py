#!/usr/bin/env python3
"""
Publish mini-blog.md as a new Discussion on the Hugging Face Space.

The public HF MCP in Cursor is read-only; Discussions are the supported
write path via huggingface_hub (not the same as a "Community article", but
visible on the Space Discussions tab).

Usage:
  export HF_TOKEN=hf_...   # or: huggingface-cli login
  python scripts/publish_blog_to_hub_discussion.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "vikramsrini/sprint_planning_env"
TITLE = "SprintBoard — blog: training LLMs to plan sprints (OpenEnv #3.1)"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    body = (root / "mini-blog.md").read_text(encoding="utf-8")
    if not body.strip():
        print("empty mini-blog.md", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "Set HF_TOKEN (https://huggingface.co/settings/tokens) or run "
            "`huggingface-cli login` first.",
            file=sys.stderr,
        )
        return 1

    api = HfApi(token=token)
    d = api.create_discussion(
        repo_id=REPO,
        title=TITLE,
        description=body,
        repo_type="space",
    )
    print("Created discussion:", d.url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
