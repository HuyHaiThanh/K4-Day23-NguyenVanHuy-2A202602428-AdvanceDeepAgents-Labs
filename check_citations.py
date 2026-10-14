"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys
from collections import Counter
from urllib.parse import urlsplit

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def without_code(text):
    """Mask code while preserving character offsets, for locating real headings."""
    lines, fence = [], None
    def blank(line):
        return re.sub(r"[^\r\n]", " ", line)
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1] and not marker[2].strip():
                fence = None
            lines.append(blank(line))
        elif marker:
            fence = (marker[1][0], len(marker[1]))
            lines.append(blank(line))
        elif line.startswith(("    ", "\t")):
            lines.append(blank(line))
        else:
            lines.append(line)
    result = "".join(lines)
    return re.sub(r"(`+)([^`]|(?!\1)`)*?\1", lambda m: blank(m[0]), result)


def prose_only(text):
    """Ignore Markdown link labels as well as code when counting citations."""
    return re.sub(r"\[[^\]\n]*\]\([^\n]*?\)", "", without_code(text))


def valid_url(value):
    """Require an absolute HTTP(S) URL without whitespace."""
    if not isinstance(value, str) or re.search(r"\s", value):
        return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except ValueError:
        return False


def audit_sample_numbers(sources, count=5):
    """Deterministic samples span the final bibliography; structural checks still cover all."""
    numbers = sorted(s["n"] for s in sources)
    if len(numbers) <= count:
        return set(numbers)
    return {numbers[i * (len(numbers) - 1) // (count - 1)] for i in range(count)}


def citation_numbers(text):
    """Recognise [1], [1, 2] and [1-3]; bound expansion of hostile ranges."""
    numbers = set()
    for match in re.finditer(r"\[(\d[\d\s,–-]*)\]", text):
        grammar = r"\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*"
        if not re.fullmatch(grammar, match[1]) or any(len(n) > 10 for n in re.findall(r"\d+", match[1])):
            numbers.add(-1)
            continue
        for part in match[1].split(","):
            if "-" in part or "–" in part:
                start, end = map(int, re.split(r"[-–]", part))
                if start <= end <= start + 10000:
                    numbers.update(range(start, end + 1))
                else:
                    numbers.add(-1)
            else:
                numbers.add(int(part))
    return numbers


def source_metadata_problems(sources, require_metadata=False):
    """Check research-record metadata when supplied, returning ALL errors for sandbox repair."""
    problems = []
    for source in sources:
        if not isinstance(source, dict) or (not require_metadata and "source" not in source):
            continue
        label, paper_id, url = source.get("source"), source.get("id"), source.get("url")
        prefix = f"source [{source.get('n')}] {label}"
        if not isinstance(label, str) or label not in {"arxiv", "hf-daily", "hf-search", "web"}:
            problems.append(prefix + ": unknown source family; use actual retrieval tool provenance")
            continue
        for key in ("id", "title", "date"):
            if not isinstance(source.get(key), str):
                problems.append(prefix + f": {key} must be a string (date may be empty)")
        if isinstance(source.get("title"), str) and not source["title"].strip():
            problems.append(prefix + ": title is empty")
        if not isinstance(paper_id, str):
            continue
        if label == "arxiv" and (not paper_id or re.search(r"v\d+$", paper_id) or url != f"https://arxiv.org/abs/{paper_id}"):
            problems.append(prefix + f": id/URL mismatch (id={paper_id!r}, URL={url!r}); "
                            "use the retrieved canonical arXiv id and HTTPS /abs/ URL, without vN")
        if label in {"hf-daily", "hf-search"} and (not paper_id or url != f"https://huggingface.co/papers/{paper_id}"):
            problems.append(prefix + f": id/URL mismatch (id={paper_id!r}, URL={url!r}); "
                            "use paper.id and https://huggingface.co/papers/<paper.id> from the actual HF result, "
                            "not its githubRepo field. Retrieve missing evidence; never invent ids or relabel sources.")
    return problems


def report_structure_problems(report, require_template=False):
    """Enforce the survey template when a survey section is present, or explicitly required."""
    headings = re.findall(r"^##[ \t]+(.+?)[ \t]*\r?$", without_code(report), re.M)
    fixed = {"TL;DR", "Background", "Trends and open problems", "References"}
    if not require_template and not any(h.lower() in {s.lower() for s in fixed - {"References"}} for h in headings):
        return []  # A generic citation fixture need not be a complete survey.
    problems = []
    canonical = {h.lower(): h for h in fixed}
    for heading in headings:
        if heading.lower() in canonical and heading != canonical[heading.lower()]:
            problems.append(f"non-canonical heading: {heading}; use exactly {canonical[heading.lower()]}")
    for heading in sorted(fixed - {"References"}):
        if headings.count(heading) != 1:
            problems.append(f"expected exactly one required report section: {heading} (exact spelling/case)")
    themes = [h for h in headings if h.lower() not in canonical]
    if not 3 <= len(themes) <= 6:
        problems.append(f"report needs 3-6 thematic sections (found {len(themes)}: {', '.join(themes)}); "
                        "merge related themes; do not add other level-2 sections")
    body = re.split(r"^##[ \t]+References[ \t]*\r?$", prose_only(report), maxsplit=1, flags=re.M)[0]
    for line in body.splitlines():
        if re.match(r"^\s*\[\d+\].*https?://", line):
            problems.append("bibliography entry outside References: " + line[:100])
        bullet = re.match(r"^\s*[-*+]\s+(.+)$", line)
        if bullet and len(bullet[1].split()) >= 3 and not citation_numbers(bullet[1]):
            problems.append("survey bullet needs its own evidence citation: " + bullet[1][:100])
    for match in re.finditer(r"\[\d+\]\.[A-Za-z]", body):
        problems.append("text appears corrupted after a citation (missing sentence boundary): " +
                        body[match.start():match.start() + 100] + "; rewrite the complete affected paragraph")
    for paragraph in re.split(r"\n\s*\n", body):
        prose = " ".join(line for line in paragraph.splitlines() if line.strip() and not line.lstrip().startswith("#"))
        if len(prose.split()) >= 10 and not citation_numbers(prose):
            problems.append("survey prose paragraph needs an evidence citation: " + prose[:100])
    return problems


def check(report_text, sources):
    """Return structural citation problems; factual support requires a separate audit."""
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json (expected a non-empty array)"]
    problems, indexed, urls = [], {}, set()
    for position, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            problems.append(f"source entry {position} must be an object")
            continue
        n, url = source.get("n"), source.get("url")
        if type(n) is not int or n <= 0:
            problems.append(f"source entry {position}: n must be a positive integer")
        elif n in indexed:
            problems.append(f"duplicate source number [{n}]")
        else:
            indexed[n] = source
        if not valid_url(url):
            problems.append(f"source entry {position}: invalid HTTP(S) URL")
        elif url in urls:
            problems.append(f"duplicate source URL: {url}")
        else:
            urls.add(url)
    problems.extend(source_metadata_problems(sources))
    if not isinstance(report_text, str) or not report_text.strip():
        return problems + ["report is empty or not text"]
    problems.extend(report_structure_problems(report_text))
    # Do not mask Markdown links in References: those URLs still need checking.
    headings = list(re.finditer(r"^##[ \t]+References[ \t]*\r?$", without_code(report_text), re.M))
    if not headings:
        return problems + ["missing ## References heading"]
    if len(headings) != 1:
        problems.append("expected exactly one ## References heading")
    body, references = report_text[:headings[0].start()], report_text[headings[0].end():]
    cited = citation_numbers(prose_only(body))
    if -1 in cited:
        problems.append("invalid citation group or range")
        cited.remove(-1)
    for n in sorted(cited - indexed.keys()):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(indexed.keys() - cited):
        problems.append(f"source [{n}] never cited")
    counts = Counter()
    for line in references.splitlines():
        match = re.match(r"^\[(\d+)\]\s*(.*)$", line)
        if not match:
            continue
        n = int(match[1])
        counts[n] += 1
        if n not in indexed:
            problems.append(f"reference [{n}] is not a source")
            continue
        found = re.findall(r"https?://[^\s<>\]\"`]+", match[2])
        cleaned = []
        for url in found:
            url = url.rstrip(".,;")
            while url.endswith(")") and url.count(")") > url.count("("):
                url = url[:-1]
            cleaned.append(url)
        if len(cleaned) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL")
        elif cleaned[0] != indexed[n].get("url"):
            problems.append(f"reference [{n}] URL does not match sources.json")
    for n in sorted(indexed):
        if counts[n] != 1:
            problems.append(f"source [{n}] needs exactly one reference line (found {counts[n]})")
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
