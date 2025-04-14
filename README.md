# repo-qa

Ask questions about any local git repository in plain English. Built on LangChain
and Claude, with local embeddings (no extra API keys) and a FAISS vector store.

```
$ repo-qa index ~/code/some-project
Indexing /Users/me/code/some-project…
Indexed 1842 chunks. Index saved.

$ repo-qa ask ~/code/some-project "Where is authentication handled?"
Authentication is handled by the middleware in `src/auth/middleware.ts:14-58`,
which reads the `session` cookie and verifies it via `verifySession()` defined
in `src/auth/session.ts:22-40`. Routes that require auth use the
`requireUser()` helper from `src/auth/guards.ts:8-25`. …
```

## Why this is useful

- **Onboarding to a new codebase.** Ask "where is the payment flow?" instead of
  fishing through `grep`.
- **Reading code at distance.** Get a synthesized answer with file:line citations
  pointing at the exact lines, so you can verify before trusting.
- **Cheap and local.** Embeddings run locally via FastEmbed; only the final
  answer call hits Anthropic.

## How it works (LangChain showcase)

- **`git ls-files`** to list every tracked file (respects `.gitignore` for free).
- **`RecursiveCharacterTextSplitter.from_language()`** to split each file with
  syntax-aware boundaries (Python, TS, Go, Rust, …).
- **`FastEmbedEmbeddings`** (BGE small) for local embeddings — no API key.
- **`FAISS`** vector store, persisted to `~/.cache/repo-qa/<hash>`.
- **LCEL chain**: retriever → prompt → `ChatAnthropic` → string parser.

## Install

```bash
cd repo-qa
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env  # then put your ANTHROPIC_API_KEY in .env
```

## Use

```bash
# One-time per repo (or whenever it changes significantly):
repo-qa index /path/to/repo

# Single question:
repo-qa ask /path/to/repo "How does the rate limiter work?"

# Or open a REPL:
repo-qa shell /path/to/repo
```

The first index of a given repo will download the embedding model (~130 MB).
Subsequent indexes and queries are fast.

## Tuning

Edit `repo_qa/config.py`:

- `DEFAULT_CHAT_MODEL` — Claude model to answer with (default: Haiku 4.5).
- `TOP_K` — how many code excerpts to retrieve per question (default: 6).
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — splitter granularity.
- `MAX_FILE_BYTES` — files larger than this are skipped (200 KB default).
