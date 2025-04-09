"""Command-line interface for repo-qa."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import __version__


def cmd_index(args: argparse.Namespace) -> int:
    # Lazy import — keeps `--help` / `--version` snappy without loading FAISS/fastembed.
    from .indexer import build_index

    repo = Path(args.path).resolve()
    print(f"Indexing {repo}…")
    store = build_index(repo)
    print(f"Indexed {store.index.ntotal} chunks. Index saved.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .qa import ask

    repo = Path(args.path).resolve()
    print(ask(repo, args.question))
    return 0


def cmd_shell(args: argparse.Namespace) -> int:
    from .qa import build_qa_chain

    repo = Path(args.path).resolve()
    chain = build_qa_chain(repo)
    print(f"repo-qa shell — {repo}")
    print("Type a question, or Ctrl-D / Ctrl-C to exit.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        try:
            print(chain.invoke(q))
        except Exception as e:  # noqa: BLE001 — REPL: surface and keep going
            print(f"error: {e}", file=sys.stderr)
        print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="repo-qa",
        description="Ask questions about a local git repository using LangChain + Claude.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Index a repository")
    p_index.add_argument("path", help="Path to the git repo")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("path", help="Path to the git repo")
    p_ask.add_argument("question", help="Your question")
    p_ask.set_defaults(func=cmd_ask)

    p_shell = sub.add_parser("shell", help="Interactive Q&A REPL")
    p_shell.add_argument("path", help="Path to the git repo")
    p_shell.set_defaults(func=cmd_shell)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
