"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", str(topic).strip().lower()).strip("-")
    if not cleaned:
        cleaned = "topic"
    return cleaned[:60].rstrip("-")


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return f"""Please conduct a rigorous, comprehensive academic deep research survey on the following topic:
"{topic}"

Follow these required phases in order:
1. PLANNING: Use `write_todos` to create an actionable plan with at least 3 thematic sub-questions.
2. DELEGATION (Subagent calls >= 3):
   You MUST make at least 3 distinct researcher delegation calls using the `task` tool to ensure complete literature and platform coverage:
   - Task 1: Delegate sub-question 1 focusing on foundational architectures and theory to `researcher` using `arxiv_search`. (Produces "source": "arxiv", url: https://arxiv.org/abs/<id>)
   - Task 2: Delegate sub-question 2 focusing on trending models, empirical results, and open-source releases to `researcher` using `hf_daily_papers` (with appropriate keyword). (Produces "source": "hf-daily", url: https://huggingface.co/papers/<id>)
   - Task 3: Delegate sub-question 3 focusing on benchmarks, fine-tuning, and specialized methods to `researcher` using `hf_search_papers`. (Produces "source": "hf-search", url: https://huggingface.co/papers/<id>)
   Give full context in each task call: topic, sub-question, target notes path under `/tmp/work/research/notes/`, and the required tools/source family.
3. CONSOLIDATE SOURCES & VERIFY DIVERSITY (RUBRIC 2.2):
   Read the notes files, extract all verified sources, and write them to `{SOURCES_PATH}` as a JSON list of objects:
   [
     {{"n": 1, "id": "<id>", "url": "<url>", "title": "<title>", "date": "<YYYY-MM-DD>", "source": "arxiv" | "hf-daily" | "hf-search" | "web"}},
     ...
   ]
   CRITICAL REQUIREMENT: `sources.json` MUST contain at least 3 distinct source families among "arxiv", "hf-daily", "hf-search", "web" (for example, papers from arxiv, hf-daily, and hf-search).
   Check your list of sources before proceeding: if any family is missing so that you have fewer than 3 families, IMMEDIATELY call `researcher` to gather sources for that missing family!
4. WRITE REPORT BODY:
   Write the full survey report in English to `{REPORT_PATH}` according to REPORT_TEMPLATE.md.
   Include: `# Title`, `## TL;DR` (with citations [n]), `## Background` (with citations [n]), 3-6 comparative thematic sections (with inline citations [n]), and `## Trends and open problems` (with citations [n]).
   Cite sources from all collected families across the report. DO NOT write `## References` manually.
5. FINALIZE CITATIONS:
   Run `python3 {FINALIZER_PATH}` using the `execute` shell tool to generate references and renumber citations.
6. VALIDATE CITATIONS:
   Run `python3 {VALIDATOR_PATH}` using the `execute` shell tool. Verify that it prints `OK: ...`.
7. FACT CHECK:
   Delegate to `citation-checker` using `task` to spot-check 3-4 claims with their source URLs.
"""


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_counts = Counter()
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        # Check tool_calls
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", str(tc))
                if name:
                    tool_counts[name] += 1
        elif isinstance(msg, dict) and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", str(tc))
                if name:
                    tool_counts[name] += 1

        # Check usage_metadata
        usage = getattr(msg, "usage_metadata", None)
        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
        elif isinstance(msg, dict) and "usage_metadata" in msg and isinstance(msg["usage_metadata"], dict):
            input_tokens += int(msg["usage_metadata"].get("input_tokens") or 0)
            output_tokens += int(msg["usage_metadata"].get("output_tokens") or 0)

    subagent_calls = tool_counts.get("task", 0)
    return {
        "model": str(model_name),
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_counts),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError(f"Missing or empty report at {REPORT_PATH}")
    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError(f"Missing or empty sources at {SOURCES_PATH}")

    try:
        sources = json.loads(sources_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"sources.json is invalid JSON: {exc}") from exc

    if not isinstance(sources, list) or len(sources) == 0:
        raise RuntimeError("sources.json must be a non-empty list of objects")

    source_families = sorted(list({str(s.get("source")) for s in sources if s.get("source")}))
    allowed_families = {"arxiv", "hf-daily", "hf-search", "web"}
    if len(allowed_families & set(source_families)) < 3:
        raise RuntimeError(
            f"Insufficient source families in sources.json: found {source_families}, need >= 3 of {sorted(allowed_families)}"
        )

    summary = summarize(messages, elapsed, model_name)
    meta = {
        "topic": topic,
        "n_sources": len(sources),
        "source_families": source_families,
        **summary,
    }

    slug = slugify(topic)
    reports_dir.mkdir(parents=True, exist_ok=True)

    sources_dest = reports_dir / f"{slug}.sources.json"
    meta_dest = reports_dir / f"{slug}.meta.json"
    report_dest = reports_dir / f"{slug}.md"

    sources_dest.write_bytes(sources_bytes)
    meta_dest.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    report_dest.write_bytes(report_bytes)

    return report_dest


def _rotate_api_key():
    """Rotate to the next configured Gemini API key if available."""
    keys = [
        os.getenv("GOOGLE_API_KEY", ""),
        os.getenv("GOOGLE_API_KEY_2", ""),
        os.getenv("GOOGLE_API_KEY_3", ""),
    ]
    keys = [k.strip() for k in keys if k.strip()]
    current = os.getenv("GOOGLE_API_KEY", "").strip()
    if not keys:
        return False
    try:
        idx = keys.index(current)
        next_idx = (idx + 1) % len(keys)
    except ValueError:
        next_idx = 0

    next_key = keys[next_idx]
    if next_key != current:
        print(f"[research] Rotating GOOGLE_API_KEY to key #{next_idx + 1}", file=sys.stderr)
        os.environ["GOOGLE_API_KEY"] = next_key
        return True
    return False


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    if not topic or not topic.strip():
        print("Usage: python research.py <topic>", file=sys.stderr)
        return 2

    clean_topic = topic.strip()
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            model = make_model()
            start = time.monotonic()
            with open_sandbox() as backend:
                backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
                upload(backend, {
                    VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                    FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                })
                agent = build_lead_agent(backend, model)
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": build_prompt(clean_topic)}]},
                    config={"recursion_limit": 1000},
                )
                messages = result.get("messages", [])
                elapsed = time.monotonic() - start
                model_name = (
                    getattr(model, "model_name", None)
                    or getattr(model, "model", None)
                    or os.getenv("LAB_MODEL", "model")
                )
                report_path = save_outputs(backend, clean_topic, messages, elapsed, model_name)
                print(f"SUCCESS: Report saved to {report_path}")
                return 0
        except Exception as exc:
            err_msg = str(exc)
            print(f"[research] Attempt {attempt + 1} failed: {type(exc).__name__}: {err_msg}", file=sys.stderr)
            # If error relates to quota / rate limit / 429, try rotating key
            if ("429" in err_msg or "ResourceExhausted" in err_msg or "quota" in err_msg.lower()) and attempt + 1 < max_attempts:
                if _rotate_api_key():
                    time.sleep(2)
                    continue
            if attempt + 1 == max_attempts:
                print(f"FAILED: {exc}", file=sys.stderr)
                return 1
            time.sleep(3)

    return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
