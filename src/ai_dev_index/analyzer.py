from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from ai_dev_index.config import Settings

logger = logging.getLogger(__name__)

# Curated AI software engineering entity lexicon for deterministic topic extraction
# and canonical normalization across videos.
ENTITY_PATTERNS: list[tuple[str, str, str]] = [
    # (Canonical Name, Type, Regex)
    ("Claude Code", "tool", r"\bclaude\s+code\b"),
    ("Claude", "model", r"\bclaude(?:\s+\d+(?:\.\d+)?|\s+(?:opus|sonnet|haiku))?\b"),
    ("GPT-5", "model", r"\bgpt[-\s]?5(?:\.\d+)?\b"),
    ("GPT-4o", "model", r"\bgpt[-\s]?4o\b"),
    ("OpenAI o3", "model", r"\b(?:openai\s+)?o3(?:-mini|-pro)?\b"),
    ("Gemini", "model", r"\bgemini(?:\s+\d+(?:\.\d+)?)?\b"),
    ("DeepSeek", "model", r"\bdeepseek(?:[-\s]?(?:r1|v3|v4))?\b"),
    ("Llama", "model", r"\bllama\s*[34]\b"),
    ("Qwen", "model", r"\bqwen(?:\s*\d+(?:\.\d+)?)?\b"),
    ("Codex", "tool", r"\bcodex\b"),
    ("Cursor", "tool", r"\bcursor(?:\s+ide|\s+composer|\s+agent)?\b"),
    ("Windsurf", "tool", r"\bwindsurf\b"),
    ("OpenCode", "tool", r"\bopencode\b"),
    ("Aider", "tool", r"\baider\b"),
    ("Cline", "tool", r"\bcline\b"),
    ("GitHub Copilot", "tool", r"\b(?:github\s+)?copilot\b"),
    ("v0", "tool", r"\bv0(?:\.dev)?\b"),
    ("MCP", "protocol", r"\b(?:mcp|model\s+context\s+protocol)\b"),
    ("AI SDK", "framework", r"\b(?:vercel\s+)?ai\s+sdk\b"),
    ("LangGraph", "framework", r"\blanggraph\b"),
    ("LangChain", "framework", r"\blangchain\b"),
    ("LangSmith", "tool", r"\blangsmith\b"),
    ("PydanticAI", "framework", r"\bpydantic\s*ai\b"),
    ("CrewAI", "framework", r"\bcrewai\b"),
    ("Ollama", "tool", r"\bollama\b"),
    ("vLLM", "infra", r"\bvllm\b"),
    ("OpenRouter", "infra", r"\bopenrouter\b"),
    ("RAG", "concept", r"\brag\b|\bretrieval[-\s]augmented\b"),
    ("Coding Agents", "concept", r"\bcoding\s+agents?\b|\bagentic\s+coding\b"),
    ("AI Agents", "concept", r"\b(?:ai|multi[-\s]?)agents?\b"),
    ("Context Engineering", "concept", r"\bcontext\s+engineering\b"),
    ("Prompt Caching", "concept", r"\bprompt\s+caching\b"),
    ("Structured Outputs", "concept", r"\bstructured\s+outputs?\b"),
    ("Evals", "concept", r"\bevals?\b|\bevaluations?\b"),
    ("Fine-Tuning", "concept", r"\bfine[-\s]?tun(?:ing|ed)\b"),
    ("Reasoning Models", "concept", r"\breasoning\s+models?\b|\btest[-\s]?time\s+compute\b"),
    ("Benchmarks", "concept", r"\b(?:swe[-\s]?bench|benchmark|benchmarks|arc[-\s]?agi)\b"),
    ("TypeScript", "language", r"\btypescript\b"),
    ("Python", "language", r"\bpython\b"),
    ("Next.js", "framework", r"\bnext\.?js\b"),
    ("Anthropic", "company", r"\banthropic\b"),
    ("OpenAI", "company", r"\bopenai\b"),
    ("Google DeepMind", "company", r"\bdeepmind\b"),
    ("Vercel", "company", r"\bvercel\b"),
    ("Cloudflare", "company", r"\bcloudflare\b"),
    ("Docker", "infra", r"\bdocker\b"),
]

COMPILED_PATTERNS = [
    (name, etype, re.compile(pattern, re.IGNORECASE))
    for name, etype, pattern in ENTITY_PATTERNS
]

CANONICAL_LOOKUP = {name.lower(): (name, etype) for name, etype, _ in ENTITY_PATTERNS}


@dataclass(slots=True)
class AnalysisResult:
    summary: str
    takeaways: list[str] = field(default_factory=list)
    key_moments: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, str]] = field(default_factory=list)
    sentiment: str = "balanced"
    status: str = "heuristic"  # llm | heuristic | error
    model: str = "heuristic-v1"
    error: str = ""


def extract_known_entities(text: str, *, max_entities: int = 12) -> list[dict[str, str]]:
    counts: list[tuple[int, str, str]] = []
    for name, etype, regex in COMPILED_PATTERNS:
        matches = regex.findall(text)
        if matches:
            counts.append((len(matches), name, etype))
    counts.sort(key=lambda item: (-item[0], item[1]))
    return [{"name": name, "type": etype} for _, name, etype in counts[:max_entities]]


def _normalize_entities(
    llm_entities: list[dict[str, Any]],
    detected_entities: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[str] = set()

    for item in llm_entities:
        if not isinstance(item, dict):
            continue
        raw_name = str(item.get("name", "")).strip()
        if not raw_name or len(raw_name) > 48:
            continue
        lower = raw_name.lower()
        if lower in CANONICAL_LOOKUP:
            canon_name, canon_type = CANONICAL_LOOKUP[lower]
        else:
            canon_name = raw_name
            canon_type = str(item.get("type", "topic")).strip().lower() or "topic"
        if canon_name.lower() not in seen:
            seen.add(canon_name.lower())
            merged.append({"name": canon_name, "type": canon_type})

    for item in detected_entities:
        key = item["name"].lower()
        if key not in seen:
            seen.add(key)
            merged.append(item)

    return merged[:14]


def _format_transcript_for_llm(
    segments: list[dict[str, Any]],
    fallback_text: str,
    *,
    max_chars: int = 24000,
) -> str:
    if segments:
        lines: list[str] = []
        bucket_start = 0
        bucket_words: list[str] = []
        for seg in segments:
            start = int(float(seg.get("start", 0)))
            text = " ".join(str(seg.get("text", "")).split())
            if not text:
                continue
            if not bucket_words:
                bucket_start = start
            bucket_words.append(text)
            if start - bucket_start >= 45:
                lines.append(f"[{bucket_start}s] {' '.join(bucket_words)}")
                bucket_words = []
        if bucket_words:
            lines.append(f"[{bucket_start}s] {' '.join(bucket_words)}")
        formatted = "\n".join(lines)
    else:
        formatted = fallback_text

    if len(formatted) <= max_chars:
        return formatted
    head = formatted[: int(max_chars * 0.72)]
    tail = formatted[-int(max_chars * 0.25) :]
    return f"{head}\n\n[... middle truncated for length ...]\n\n{tail}"


def analyze_heuristic(
    *,
    title: str,
    channel_name: str,
    description: str,
    transcript_text: str,
) -> AnalysisResult:
    combined = f"{title}\n{description}\n{transcript_text}"
    entities = extract_known_entities(combined)

    desc_lines = [
        line.strip()
        for line in description.splitlines()
        if line.strip() and not line.strip().lower().startswith(("http://", "https://"))
    ]
    if desc_lines:
        summary = " ".join(desc_lines[:3])[:380]
    elif transcript_text:
        words = transcript_text.split()[:55]
        summary = " ".join(words) + ("..." if len(words) == 55 else "")
    else:
        summary = f"Video by {channel_name}: {title} (configure LLM_API_KEY for full structured summary)."

    takeaways: list[str] = []
    if entities:
        top_names = ", ".join(e["name"] for e in entities[:6])
        takeaways.append(f"Topics & tools detected: {top_names}.")
    if transcript_text:
        wc = len(transcript_text.split())
        takeaways.append(
            f"Indexed locally ({wc:,} transcript words). Add LLM_API_KEY in .env to generate AI takeaways."
        )
    else:
        takeaways.append("Captions not yet available; indexed from title and description.")

    return AnalysisResult(
        summary=summary,
        takeaways=takeaways,
        key_moments=[],
        entities=entities,
        sentiment="balanced",
        status="heuristic",
        model="heuristic-v1",
    )


def analyze_video(
    *,
    title: str,
    channel_name: str,
    description: str,
    transcript_text: str,
    transcript_segments: list[dict[str, Any]],
    settings: Settings,
) -> AnalysisResult:
    detected = extract_known_entities(f"{title}\n{description}\n{transcript_text}")

    if not settings.llm_enabled or not (transcript_text or description):
        return analyze_heuristic(
            title=title,
            channel_name=channel_name,
            description=description,
            transcript_text=transcript_text,
        )

    condensed = _format_transcript_for_llm(transcript_segments, transcript_text)
    prompt = f"""Analyze this AI / software engineering YouTube video and return ONLY valid JSON matching this schema:
{{
  "summary": "2-3 concise sentences in fresh, original wording explaining the main point and why engineers care. Do NOT copy transcript sentences verbatim.",
  "takeaways": ["3 to 5 crisp, concrete engineering takeaways"],
  "entities": [{{"name": "Canonical tool/model/framework/concept name", "type": "model|tool|framework|protocol|concept|company|infra|language"}}],
  "key_moments": [{{"timestamp_sec": 120, "label": "Short topic label"}}],
  "sentiment": "bullish|skeptical|balanced|tutorial|news"
}}

Channel: {channel_name}
Title: {title}
Description:
{description[:1500]}

Transcript (with approximate [Xs] timestamps):
{condensed}
"""

    endpoint = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
        # ponytail: gateways like OpenCode Go require a stable session header
        # and prefer callers to identify with their own user agent.
        "User-Agent": "ai-dev-index",
    }
    if settings.llm_session_id.strip():
        headers["x-opencode-session"] = settings.llm_session_id.strip()
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior AI software engineer curating a concise, hype-free technical index. "
                    "Always output strict JSON and rewrite ideas in original, crisp engineering prose."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    if settings.llm_reasoning_effort.strip():
        payload["reasoning_effort"] = settings.llm_reasoning_effort.strip()

    try:
        with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
            resp = client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        raw_content = data["choices"][0]["message"]["content"]
        parsed = json.loads(raw_content)

        summary = str(parsed.get("summary", "")).strip()
        takeaways = [str(t).strip() for t in parsed.get("takeaways", []) if str(t).strip()][:6]
        key_moments = []
        for km in parsed.get("key_moments", [])[:6]:
            if isinstance(km, dict) and km.get("label"):
                key_moments.append(
                    {
                        "timestamp_sec": max(0, int(km.get("timestamp_sec", 0))),
                        "label": str(km["label"]).strip(),
                    }
                )
        entities = _normalize_entities(parsed.get("entities", []), detected)
        sentiment = str(parsed.get("sentiment", "balanced")).strip().lower()
        if sentiment not in {"bullish", "skeptical", "balanced", "tutorial", "news"}:
            sentiment = "balanced"

        if not summary:
            raise ValueError("LLM returned empty summary")

        return AnalysisResult(
            summary=summary,
            takeaways=takeaways,
            key_moments=key_moments,
            entities=entities,
            sentiment=sentiment,
            status="llm",
            model=settings.llm_model,
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning("LLM analysis failed for '%s': %s — falling back to heuristic", title, exc)
        fallback = analyze_heuristic(
            title=title,
            channel_name=channel_name,
            description=description,
            transcript_text=transcript_text,
        )
        fallback.error = f"LLM fallback: {exc}"
        return fallback
