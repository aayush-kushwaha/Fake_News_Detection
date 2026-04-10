"""Groq client helpers for optional LLM analysis.

This module keeps API access separate from the notebook so the core
fake-news workflow stays focused on data loading and model training.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_ARTICLE_CHARS = 4000
MAX_EVIDENCE_SNIPPET_CHARS = 400

TRUSTED_NEWS_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "cbsnews.com",
    "cnn.com",
    "npr.org",
    "nbcnews.com",
    "nytimes.com",
    "abcnews.go.com",
    "usatoday.com",
    "theguardian.com",
    "politico.com",
    "washingtonpost.com",
    "wsj.com",
}


def get_groq_client(api_key: Optional[str] = None) -> OpenAI:
    """Create a Groq-compatible OpenAI client."""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set.")

    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")


def ask_groq(
    prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """Send a prompt to Groq and return the plain text response.

    Retries are used to handle transient transport or decode failures.
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            client = get_groq_client(api_key=api_key)
            response = client.responses.create(input=prompt, model=model)
            return response.output_text
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"Groq API call failed after {max_retries} attempts: {last_error}")


def summarize_article(article_text: str, model: str = DEFAULT_MODEL, api_key: Optional[str] = None) -> str:
    """Generate a short summary of an article."""
    article_excerpt = article_text[:MAX_ARTICLE_CHARS]
    prompt = (
        "Summarize the following news article in 3 concise bullet points. "
        "Focus on the main claim, any named entities, and the overall tone.\n\n"
        f"Article:\n{article_excerpt}"
    )
    return ask_groq(prompt=prompt, model=model, api_key=api_key)


def explain_fake_news_risk(article_text: str, model: str = DEFAULT_MODEL, api_key: Optional[str] = None) -> str:
    """Ask Groq for a lightweight fake-news risk explanation.

    This does not verify facts against external sources. It only analyzes
    the article text for language cues, internal consistency, and weak
    signals that may justify a closer review.
    """
    article_excerpt = article_text[:MAX_ARTICLE_CHARS]
    prompt = (
        "You are helping assess whether a news article may be misleading. "
        "Do not claim external fact-checking. Instead, analyze the text for "
        "signs such as sensational wording, vague sourcing, emotional tone, "
        "or internal inconsistencies. Return a short, plain-English assessment.\n\n"
        f"Article:\n{article_excerpt}"
    )
    return ask_groq(prompt=prompt, model=model, api_key=api_key)


def _is_trusted_source(url: str) -> bool:
    """Return True when URL belongs to a trusted news domain."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False

    if not host:
        return False
    if host.startswith("www."):
        host = host[4:]
    return any(host == domain or host.endswith(f".{domain}") for domain in TRUSTED_NEWS_DOMAINS)


def retrieve_trusted_evidence(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search public web results and keep only trusted-source evidence.

    Requires `ddgs` package.
    Returns: list of {title, url, snippet}.
    """
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise ImportError("ddgs is required for external retrieval.") from exc

    results: List[Dict[str, str]] = []
    fallback_results: List[Dict[str, str]] = []
    seen_urls = set()
    words = query.split()
    query_variants = [
        " ".join(words[:20]),
        f"{' '.join(words[:12])} news",
        f"{' '.join(words[:12])} fact check",
    ]
    query_variants = [q.strip() for q in query_variants if q.strip()]

    with DDGS() as ddgs:
        for q in query_variants:
            # First pass: "news" index tends to surface article sources faster.
            try:
                for item in ddgs.news(q, max_results=20):
                    url = str(item.get("url") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    title = str(item.get("title") or "").strip()
                    snippet = str(item.get("body") or "").strip()
                    seen_urls.add(url)
                    entry = {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "source_type": "trusted" if _is_trusted_source(url) else "other",
                    }
                    if entry["source_type"] == "trusted":
                        results.append(entry)
                        if len(results) >= max_results:
                            return results
                    else:
                        fallback_results.append(entry)
            except Exception:
                # Some queries may return no results and raise; continue to fallback queries.
                pass

            # Second pass: web search as fallback.
            try:
                for item in ddgs.text(q, max_results=20):
                    url = str(item.get("href") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    title = str(item.get("title") or "").strip()
                    snippet = str(item.get("body") or "").strip()
                    seen_urls.add(url)
                    entry = {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "source_type": "trusted" if _is_trusted_source(url) else "other",
                    }
                    if entry["source_type"] == "trusted":
                        results.append(entry)
                        if len(results) >= max_results:
                            return results
                    else:
                        fallback_results.append(entry)
            except Exception:
                pass

    if results:
        return results
    return fallback_results[:max_results]


def fact_check_with_external_references(
    article_text: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_results: int = 5,
) -> Dict[str, Any]:
    """Run a lightweight fact check using external retrieved sources.

    Returns a dict with:
    - verdict: supported | contradicted | mixed | insufficient_evidence
    - confidence: integer 0-100
    - explanation: short text
    - references: list of cited {title, url, snippet}
    - raw_model_output: model output when JSON parsing fails
    """
    article_excerpt = article_text[:MAX_ARTICLE_CHARS]
    search_query = " ".join(article_excerpt.split()[:40])
    try:
        evidence = retrieve_trusted_evidence(search_query, max_results=max_results)
    except Exception as exc:
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0,
            "explanation": f"External retrieval failed: {exc}",
            "references": [],
            "raw_model_output": None,
            "has_trusted_references": False,
        }

    if not evidence:
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0,
            "explanation": "No external evidence was retrieved for this article.",
            "references": [],
            "raw_model_output": None,
        }

    has_trusted = any(ev.get("source_type") == "trusted" for ev in evidence)

    evidence_lines = []
    for idx, ev in enumerate(evidence, start=1):
        snippet = (ev.get("snippet") or "")[:MAX_EVIDENCE_SNIPPET_CHARS]
        evidence_lines.append(
            f"[{idx}] {ev['title']}\nURL: {ev['url']}\nsource_type: {ev.get('source_type', 'other')}\nSnippet: {snippet}"
        )
    evidence_block = "\n\n".join(evidence_lines)

    prompt = (
        "You are a cautious fact-checking assistant. Use only the provided external references. "
        "Do not invent sources. If evidence is weak, return insufficient_evidence. "
        "Treat references marked source_type=other as lower reliability than trusted sources.\n\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        "  \"verdict\": \"supported|contradicted|mixed|insufficient_evidence\",\n"
        "  \"confidence\": 0-100,\n"
        "  \"explanation\": \"short explanation\",\n"
        "  \"reference_ids\": [1,2]\n"
        "}\n\n"
        f"Article to verify:\n{article_excerpt}\n\n"
        f"External references:\n{evidence_block}"
    )

    raw = ask_groq(prompt=prompt, model=model, api_key=api_key)

    try:
        parsed = json.loads(raw)
        ref_ids = parsed.get("reference_ids", [])
        refs = []
        for rid in ref_ids:
            if isinstance(rid, int) and 1 <= rid <= len(evidence):
                refs.append(evidence[rid - 1])

        if not refs:
            refs = evidence[: min(2, len(evidence))]

        return {
            "verdict": parsed.get("verdict", "insufficient_evidence"),
            "confidence": int(parsed.get("confidence", 0)),
            "explanation": str(parsed.get("explanation", "No explanation provided.")),
            "references": refs,
            "raw_model_output": raw,
            "has_trusted_references": has_trusted,
        }
    except Exception:
        return {
            "verdict": "insufficient_evidence",
            "confidence": 0,
            "explanation": "Could not parse model output into structured fact-check JSON.",
            "references": evidence[: min(2, len(evidence))],
            "raw_model_output": raw,
            "has_trusted_references": has_trusted,
        }


def format_fact_check_report(result: Dict[str, Any]) -> str:
    """Format fact-check results with explicit source references."""
    lines = [
        f"Verdict: {result.get('verdict', 'insufficient_evidence')}",
        f"Confidence: {result.get('confidence', 0)}",
        f"Explanation: {result.get('explanation', '')}",
        "References:",
    ]
    refs = result.get("references", [])
    if not refs:
        lines.append("- None")
    else:
        for ref in refs:
            source_tag = ref.get("source_type", "other")
            lines.append(f"- [{source_tag}] {ref.get('title', '')}: {ref.get('url', '')}")
    if not result.get("has_trusted_references", False):
        lines.append("Note: No trusted-source references were found; treat verdict as low confidence.")
    return "\n".join(lines)


if __name__ == "__main__":
    sample_text = (
        "Sample headline: Local council approves new transport plan after heated debate. "
        "The article describes budget changes, public reaction, and next steps."
    )
    print(explain_fake_news_risk(sample_text))