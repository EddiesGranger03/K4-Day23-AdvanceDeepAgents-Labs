"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]  [2-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            try:
                numbers.append(int(part))
            except ValueError:
                pass
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    Rules:
      1. sources must be a non-empty list of objects.
      2. Each source entry:
         - n must be an int
         - url must start with http:// or https://
         - the same url must not appear twice in sources.json
      3. Report must contain heading '## References'. Text before it is body.
      4. Every [n] cited in body must exist in sources.json; every source must be cited at least once.
      5. '## References' section has exactly one reference line per source, starting with '[n]'.
      6. Each reference line contains exactly one HTTP(S) URL equal to that source's url.
    """
    problems = []

    if not isinstance(sources, list) or len(sources) == 0:
        return ["no sources in sources.json"]

    source_by_n = {}
    seen_urls = {}
    for idx, s in enumerate(sources):
        if not isinstance(s, dict):
            problems.append(f"source index {idx} is not an object")
            continue
        n = s.get("n")
        if not isinstance(n, int):
            problems.append(f"source entry {idx} has non-integer n: {n!r}")
            continue
        if n in source_by_n:
            problems.append(f"duplicate source number n={n} in sources.json")
        source_by_n[n] = s

        url = s.get("url")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] has invalid or missing url: {url!r}")
        else:
            canonical_url = url.strip()
            if canonical_url in seen_urls:
                problems.append(f"duplicate url {canonical_url!r} shared by source [{seen_urls[canonical_url]}] and [{n}]")
            else:
                seen_urls[canonical_url] = n

    source_numbers = set(source_by_n.keys())

    ref_matches = list(_REF_HEADING.finditer(report_text))
    if not ref_matches:
        problems.append("missing heading '## References' in report")
        body = report_text
        ref_section = ""
    else:
        last_match = ref_matches[-1]
        body = report_text[:last_match.start()]
        ref_section = report_text[last_match.end():]

    # Extract [n] from body only (ignoring code blocks and markdown links)
    segments = _CODE.split(body)
    cited_numbers = set()
    for i, segment in enumerate(segments):
        if i % 2 == 1:
            continue  # code block
        for match in _GROUP.finditer(segment):
            for n in _group_numbers(match.group(1)):
                cited_numbers.add(n)

    # Citations in body vs sources.json
    for n in sorted(cited_numbers):
        if n not in source_numbers:
            problems.append(f"[{n}] cited but missing from sources.json")

    for n in sorted(source_numbers):
        if n not in cited_numbers:
            problems.append(f"source [{n}] never cited")

    # Check References section
    if ref_matches:
        ref_lines = [line.strip() for line in ref_section.splitlines() if line.strip()]
        seen_ref_n = set()
        for line in ref_lines:
            m = re.match(r"^\[(\d+)\](?:\s+(.*))?$", line)
            if not m:
                problems.append(f"reference line does not start with '[n]': {line[:80]}")
                continue
            ref_n = int(m.group(1))
            line_rest = m.group(2) or ""

            if ref_n in seen_ref_n:
                problems.append(f"duplicate reference line for [{ref_n}]")
            seen_ref_n.add(ref_n)

            if ref_n not in source_numbers:
                problems.append(f"reference [{ref_n}] exists in References but is not in sources.json")
                continue

            urls = re.findall(r"https?://[^\s()]+", line_rest)
            if len(urls) == 0:
                problems.append(f"reference line [{ref_n}] contains no URL")
            elif len(urls) > 1:
                problems.append(f"reference line [{ref_n}] bundles multiple URLs: {urls}")
            else:
                found_url = urls[0].rstrip(".,;")
                expected_url = (source_by_n[ref_n].get("url") or "").strip().rstrip(".,;")
                if found_url != expected_url:
                    problems.append(
                        f"reference line [{ref_n}] URL mismatch: found '{found_url}', expected '{expected_url}'"
                    )

        for n in sorted(source_numbers):
            if n not in seen_ref_n:
                problems.append(f"source [{n}] is missing from References section")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
