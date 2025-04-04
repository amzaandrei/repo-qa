"""Retrieval-augmented question answering over an indexed repository."""

from __future__ import annotations

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .config import DEFAULT_CHAT_MODEL, TOP_K
from .indexer import load_index


SYSTEM_PROMPT = (
    "You are a careful code-reading assistant. Answer questions about the user's "
    "repository using only the supplied excerpts. If the answer cannot be derived "
    "from the excerpts, say so plainly — do not invent file names or APIs. "
    "Cite specific files and line ranges (e.g. `path/to/file.py:42-58`) for any "
    "claim about the code."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Question: {question}\n\nRelevant excerpts:\n{context}\n\n"
            "Write a clear, direct answer with citations.",
        ),
    ]
)


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        m = d.metadata
        header = f"{m.get('source', '?')}:{m.get('line_start', '?')}-{m.get('line_end', '?')}"
        parts.append(f"--- {header} ---\n{d.page_content}")
    return "\n\n".join(parts)


def build_qa_chain(repo_path: Path, *, model: str = DEFAULT_CHAT_MODEL, k: int = TOP_K):
    store = load_index(repo_path)
    retriever = store.as_retriever(search_kwargs={"k": k})
    llm = ChatAnthropic(model=model, temperature=0)
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


def ask(
    repo_path: Path,
    question: str,
    *,
    model: str = DEFAULT_CHAT_MODEL,
    k: int = TOP_K,
) -> str:
    return build_qa_chain(repo_path, model=model, k=k).invoke(question)
