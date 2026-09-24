#!/usr/bin/env python3
"""
Save text to a Markdown file.

Examples:
  # Type or paste text, then end input (Windows: Ctrl+Z then Enter; Unix: Ctrl+D)
  python save_markdown.py notes.md

  # Pipe content
  echo "# Title" | python save_markdown.py out.md

  # Read from an existing file and write as .md
  python save_markdown.py out.md --input raw.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Save text to a Markdown (.md) file.")
    p.add_argument(
        "output",
        type=Path,
        help="Path to the output .md file (created or overwritten).",
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Read text from this file instead of stdin.",
    )
    p.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append to the file instead of overwriting.",
    )
    p.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for read/write (default: utf-8).",
    )
    args = p.parse_args()

    if args.input is not None:
        text = args.input.read_text(encoding=args.encoding)
    else:
        if sys.stdin.isatty():
            print(
                "Reading from stdin. Paste your Markdown, then press Ctrl+Z and Enter (Windows) "
                "or Ctrl+D (macOS/Linux) to finish.",
                file=sys.stderr,
            )
        text = sys.stdin.read()

    mode = "a" if args.append else "w"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding=args.encoding, newline="")
    print(f"Wrote {len(text)} characters to {args.output.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
