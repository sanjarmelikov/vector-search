"""Grounded answers: the "G" in RAG.

The top retrieved passages go to an LLM as numbered sources, with the
instruction to answer only from them and cite each claim as [n]. We then
check the citations: a number that doesn't match a source is dropped and
reported, because an answer citing a source that doesn't exist is exactly the
kind of hallucination this pipeline exists to prevent.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

SYSTEM_PROMPT = (
    "You answer questions about scientific claims using only the numbered sources provided. "
    "Cite every factual statement with its source number in square brackets, like [1] or [2][3]. "
    "If the sources don't contain enough information, say so plainly instead of guessing. "
    "Be concise: at most 5 sentences."
)
DEFAULT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Source:
    doc_id: str
    title: str
    text: str


@dataclass
class Answer:
    text: str
    cited: list[Source]  # sources actually cited, in order of first citation
    invalid_citations: list[int] = field(default_factory=list)  # numbers with no matching source
    model: str = ""


def build_prompt(question: str, sources: list[Source]) -> str:
    blocks = [f"[{i}] {s.title}\n{s.text}" for i, s in enumerate(sources, start=1)]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def parse_citations(text: str, sources: list[Source]) -> tuple[list[Source], list[int]]:
    cited, invalid, seen = [], [], set()
    for match in _CITATION.finditer(text):
        n = int(match.group(1))
        if n in seen:
            continue
        seen.add(n)
        if 1 <= n <= len(sources):
            cited.append(sources[n - 1])
        else:
            invalid.append(n)
    return cited, invalid


class OpenAIAnswerer:
    def __init__(self, model: str = DEFAULT_MODEL, client=None):
        if client is None:
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv()  # reads OPENAI_API_KEY from .env
            client = OpenAI(max_retries=6)
        self._client = client
        self.model = model

    def answer(self, question: str, sources: list[Source]) -> Answer:
        if not sources:
            return Answer("No sources were found for this question.", [], model=self.model)
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,  # same question + sources → (nearly) the same answer
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, sources)},
            ],
        )
        text = response.choices[0].message.content.strip()
        cited, invalid = parse_citations(text, sources)
        return Answer(text, cited, invalid, self.model)
