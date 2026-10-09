"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

# Keep provider keys on the host. The sandbox never receives this environment.
load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_SEARCH_URL = "https://api.exa.ai/search"
EXA_CONTENTS_URL = "https://api.exa.ai/contents"
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}

_LAST_ARXIV_CALL = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt + 1 == attempts:
                raise
            if exc.retry_after is not None:
                delay = min(cap, max(0.0, exc.retry_after))
            else:
                ceiling = min(cap, base * (2 ** attempt))
                delay = random.uniform(0.0, ceiling)
            time.sleep(delay)


def _retry_after_seconds(value):
    """Parse Retry-After as seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _exa_post(url, payload):
    """POST to an Exa endpoint with host-side auth and bounded transient retries."""
    headers = {"accept": "application/json", "content-type": "application/json"}
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not configured; set it in the host environment or .env")
    headers["x-api-key"] = api_key

    def request():
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        except httpx.TransportError as exc:
            raise RetryableError(str(exc)) from exc

        if response.status_code in RETRYABLE_HTTP_STATUS:
            raise RetryableError(
                f"Exa returned HTTP {response.status_code}",
                retry_after=_retry_after_seconds(response.headers.get("Retry-After")),
            )
        response.raise_for_status()
        return response.json()

    return with_retry(request, attempts=5, base=1.0, cap=60.0)


def _exa_error(exc):
    """Format an error for the agent without disclosing a configured API key."""
    message = f"{type(exc).__name__}: {exc}"
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return message


# ---- TODO 2: arXiv ----
def _parse_arxiv(xml_text):
    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    records = []
    for entry in root.findall("atom:entry", ns):
        raw_id = (entry.findtext("atom:id", "", ns) or "").strip()
        paper_id = raw_id.split("/abs/")[-1].strip()
        paper_id = re.sub(r"v\d+$", "", paper_id)
        if not paper_id:
            continue
        url = f"https://arxiv.org/abs/{paper_id}"
        published = (entry.findtext("atom:published", "", ns) or "").strip()[:10]
        raw_title = entry.findtext("atom:title", "", ns) or ""
        title = " ".join(raw_title.split())
        raw_summary = entry.findtext("atom:summary", "", ns) or ""
        summary = " ".join(raw_summary.split())[:600]
        records.append({
            "id": paper_id,
            "url": url,
            "published": published,
            "title": title,
            "summary": summary,
        })
    return records


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    if not query or not query.strip():
        return "NO RESULTS"
    terms = re.findall(r"\w+", query)
    if not terms:
        return "NO RESULTS"

    try:
        count = max(1, min(30, int(max_results)))
        search_query = " AND ".join(f"all:{t}" for t in terms)

        def fetch():
            global _LAST_ARXIV_CALL
            now = time.monotonic()
            wait_time = 3.1 - (now - _LAST_ARXIV_CALL)
            if wait_time > 0:
                time.sleep(wait_time)
            _LAST_ARXIV_CALL = time.monotonic()

            try:
                headers = {"User-Agent": "DeepResearchAgent/1.0 (academic research)"}
                response = httpx.get(
                    ARXIV_URL,
                    params={
                        "search_query": search_query,
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                        "max_results": count,
                    },
                    headers=headers,
                    timeout=30.0,
                )
            except httpx.TransportError as exc:
                raise RetryableError(str(exc)) from exc

            if response.status_code in RETRYABLE_HTTP_STATUS:
                raise RetryableError(
                    f"arXiv returned HTTP {response.status_code}",
                    retry_after=_retry_after_seconds(response.headers.get("Retry-After")),
                )
            response.raise_for_status()
            return response.text

        xml_text = with_retry(fetch, attempts=5, base=2.0, cap=60.0)
        records = _parse_arxiv(xml_text)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        count = max(1, min(100, int(limit)))
        params = {"limit": count}
        if date and date.strip():
            params["date"] = date.strip()

        def fetch():
            try:
                response = httpx.get(HF_DAILY_URL, params=params, timeout=30.0)
            except httpx.TransportError as exc:
                raise RetryableError(str(exc)) from exc
            if response.status_code in RETRYABLE_HTTP_STATUS:
                raise RetryableError(
                    f"Hugging Face returned HTTP {response.status_code}",
                    retry_after=_retry_after_seconds(response.headers.get("Retry-After")),
                )
            response.raise_for_status()
            return response.json()

        data = with_retry(fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list):
            return "NO RESULTS"

        records = []
        kw = keyword.strip().lower() if keyword else ""
        for item in data:
            paper = item.get("paper") if isinstance(item, dict) and isinstance(item.get("paper"), dict) else item
            if not isinstance(paper, dict) or not paper.get("id"):
                continue
            paper_id = str(paper["id"]).strip()
            url = f"https://huggingface.co/papers/{paper_id}"
            raw_title = paper.get("title") or item.get("title") or ""
            title = " ".join(str(raw_title).split())
            raw_summary = paper.get("summary") or item.get("summary") or ""
            summary = " ".join(str(raw_summary).split())[:600]
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            upvotes = int(paper.get("upvotes") or item.get("upvotes") or 0)
            github = paper.get("githubRepo") or item.get("githubRepo") or None
            stars = paper.get("githubStars") or item.get("githubStars") or None
            if stars is not None:
                try:
                    stars = int(stars)
                except ValueError:
                    pass

            if kw and kw not in (title + " " + summary).lower():
                continue

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        records.sort(key=lambda r: r.get("upvotes", 0), reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    if not query or not query.strip():
        return "NO RESULTS"
    try:
        count = max(1, min(50, int(limit)))
        params = {"q": query.strip(), "limit": count}

        def fetch():
            try:
                response = httpx.get(HF_SEARCH_URL, params=params, timeout=30.0)
            except httpx.TransportError as exc:
                raise RetryableError(str(exc)) from exc
            if response.status_code in RETRYABLE_HTTP_STATUS:
                raise RetryableError(
                    f"Hugging Face returned HTTP {response.status_code}",
                    retry_after=_retry_after_seconds(response.headers.get("Retry-After")),
                )
            response.raise_for_status()
            return response.json()

        data = with_retry(fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list):
            return "NO RESULTS"

        records = []
        for item in data:
            paper = item.get("paper") if isinstance(item, dict) and isinstance(item.get("paper"), dict) else item
            if not isinstance(paper, dict) or not paper.get("id"):
                continue
            paper_id = str(paper["id"]).strip()
            url = f"https://huggingface.co/papers/{paper_id}"
            raw_title = paper.get("title") or item.get("title") or ""
            title = " ".join(str(raw_title).split())
            raw_summary = paper.get("ai_summary") or paper.get("summary") or item.get("summary") or ""
            summary = " ".join(str(raw_summary).split())[:600]
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            upvotes = int(paper.get("upvotes") or item.get("upvotes") or 0)
            github = paper.get("githubRepo") or item.get("githubRepo") or None
            stars = paper.get("githubStars") or item.get("githubStars") or None
            if stars is not None:
                try:
                    stars = int(stars)
                except ValueError:
                    pass

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through Exa's native HTTP API ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web with Exa. Returns compact JSON records with titles, URLs, dates, and highlights."""
    if not query or not query.strip():
        return "NO RESULTS"

    try:
        count = max(1, min(100, int(num_results)))
        search_objective = (objective or "").strip()
        if not search_objective:
            search_objective = f"Find useful web sources that help answer: {query.strip()}"

        response = _exa_post(EXA_SEARCH_URL, {
            "query": query.strip(),
            "objective": search_objective,
            "type": "auto",
            "numResults": count,
            "contents": {"highlights": True},
        })

        records = []
        for item in response.get("results", []):
            if not isinstance(item, dict) or not item.get("url"):
                continue
            highlights = item.get("highlights") or []
            if isinstance(highlights, str):
                highlights = [highlights]
            records.append({
                "id": item.get("id") or item["url"],
                "url": item["url"],
                "title": item.get("title") or "",
                "published": item.get("publishedDate") or item.get("published_date"),
                "highlights": highlights,
            })
        return json.dumps(records, ensure_ascii=False, separators=(",", ":")) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {_exa_error(exc)}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    if not url or not url.strip():
        return "NO RESULTS"

    requested_url = url.strip()
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    if api_key:
        try:
            response = _exa_post(EXA_CONTENTS_URL, {
                "urls": [requested_url],
                "text": True,
            })

            statuses = response.get("statuses") or []
            failed = next((
                status for status in statuses
                if status.get("id") == requested_url and status.get("status") != "success"
            ), None)
            if failed:
                detail = failed.get("status") or failed.get("error") or "content extraction failed"
                return f"ERROR: Exa could not extract {requested_url}: {detail}"

            result = next((
                item for item in response.get("results", [])
                if item.get("url") == requested_url
            ), None)
            if result is None and len(response.get("results", [])) == 1:
                result = response["results"][0]
            if not result:
                return "NO RESULTS"
            text = result.get("text") or ""
            return text[:12000] if text else "NO RESULTS"
        except Exception as exc:
            return f"ERROR: {_exa_error(exc)}"

    # Direct fetch fallback when EXA_API_KEY is not configured
    try:
        def fetch_direct():
            try:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                }
                r = httpx.get(requested_url, headers=headers, follow_redirects=True, timeout=20.0)
            except httpx.TransportError as exc:
                raise RetryableError(str(exc)) from exc
            if r.status_code in RETRYABLE_HTTP_STATUS:
                raise RetryableError(
                    f"HTTP {r.status_code}",
                    retry_after=_retry_after_seconds(r.headers.get("Retry-After")),
                )
            r.raise_for_status()
            return r.text

        raw_html = with_retry(fetch_direct, attempts=3, base=1.0, cap=15.0)
        text = re.sub(r"<[^>]+>", " ", raw_html)
        cleaned = " ".join(text.split())
        return cleaned[:12000] if cleaned else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
