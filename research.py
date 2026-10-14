"""Run one evidence-based survey and download unchanged, validated sandbox artifacts."""
import json
import hashlib
import os
import re
import sys
import tempfile
import time
import uuid
import threading
import base64
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from agents import AUDIT_PATH, FINALIZER_PATH, NOTES_DIR, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from check_citations import audit_sample_numbers, check, report_structure_problems, source_metadata_problems
from model import make_model
from sandbox import download, open_sandbox, upload
from tools import clear_fetched_pages, get_fetched_pages, get_source_receipts, restore_source_receipts, redact
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages.utils import count_tokens_approximately

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"
RECURSION_LIMIT = 1000
MAX_VALIDATION_ROUNDS = 2
FAMILIES = {"arxiv", "hf-daily", "hf-search", "web"}


class RunTokenBudget(BaseCallbackHandler):
    """Shared usage ledger for lead, specialists and compaction, including concurrent calls.

    Admission uses approximate input plus an output reserve. Actual usage is recorded on
    completion; this is a token guard, not a provider-enforced dollar billing limit.
    """
    raise_error = True

    def __init__(self, limit=250000):
        if limit < 1:
            raise ValueError("token budget must be positive")
        self.limit = limit
        self.lock = threading.Lock()
        self.pending = {}
        self.input_tokens = self.output_tokens = self.calls = 0
        self.unmetered_calls = 0
        self.pending_models = {}
        self.by_model = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        estimate = int(count_tokens_approximately([m for batch in messages for m in batch])) + 8192
        with self.lock:
            if self.unmetered_calls:
                raise RuntimeError("provider omitted token usage; further calls disabled")
            used = self.input_tokens + self.output_tokens + sum(self.pending.values())
            if used + estimate > self.limit:
                raise RuntimeError(f"shared token budget exhausted ({used}/{self.limit}); no further model calls")
            self.pending[run_id] = estimate
            self.pending_models[run_id] = (kwargs.get("invocation_params") or {}).get("model", "unknown")

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage = (response.llm_output or {}).get("token_usage", {})
        if not usage:
            if response.generations and response.generations[0]:
                message = response.generations[0][0].message
                usage = getattr(message, "usage_metadata", {}) or {}
        with self.lock:
            self.pending.pop(run_id, None)
            model_name = (response.llm_output or {}).get("model_name") or self.pending_models.pop(run_id, "unknown")
            self.pending_models.pop(run_id, None)
            self.calls += 1
            if not usage:
                self.unmetered_calls += 1
                raise RuntimeError("provider omitted token usage; further calls disabled")
            self.input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
            self.output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
            entry = self.by_model.setdefault(model_name, {"input": 0, "output": 0, "model_calls": 0})
            entry["input"] += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
            entry["output"] += int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
            entry["model_calls"] += 1
            if self.calls % 10 == 0:
                print(f"Usage checkpoint: {self.calls} model calls, {self.input_tokens} input, "
                      f"{self.output_tokens} output tokens (all agents).", flush=True)

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self.lock:
            self.pending.pop(run_id, None)
            self.pending_models.pop(run_id, None)

    def snapshot(self):
        with self.lock:
            return {"input": self.input_tokens, "output": self.output_tokens, "model_calls": self.calls,
                    "unmetered_calls": self.unmetered_calls,
                    "by_model": {k: dict(v) for k, v in self.by_model.items()},
                    "admission_limit": self.limit, "scope": "current_run_all_agents_and_compaction"}


def slugify(topic):
    """Safe lowercase filename, bounded to 60 characters, never empty."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(topic).lower()).strip("-")[:60].rstrip("-")
    return slug or "topic"


def merge_run_metrics(meta, previous_meta):
    """Combine measured phases, retaining their model and all-agent usage when available."""
    if previous_meta is None:
        return meta
    if previous_meta.get("topic") != meta["topic"]:
        raise RuntimeError("previous metadata belongs to a different topic")
    keys = ("model", "elapsed_s", "subagent_calls", "tool_calls", "tokens", "task_types", "current_run_usage")
    phase = lambda m, name: {"phase": name, **{k: m[k] for k in keys if k in m}}
    meta["phases"] = [*previous_meta.get("phases", [phase(previous_meta, "generation")]), phase(meta, "citation_review")]
    meta["elapsed_s"] = round(float(previous_meta["elapsed_s"]) + meta["elapsed_s"], 1)
    meta["subagent_calls"] += previous_meta["subagent_calls"]
    for key in ("tool_calls", "task_types", "tokens"):
        meta[key] = dict(Counter(previous_meta[key]) + Counter(meta[key]))
    return meta


def checkpoint_draft(backend, topic, messages, elapsed, model_name, usage, previous_meta=None):
    """Keep a failed generation's draft out of reports/ so recovery need not research again."""
    listing = backend.execute("python3 -c \"import json,pathlib; print(json.dumps([str(p) for p in pathlib.Path('/tmp/work/research/notes').glob('*.md')]))\"")
    notes = json.loads(listing.output) if listing.exit_code == 0 else []
    paths = [REPORT_PATH, SOURCES_PATH, AUDIT_PATH, *notes]
    files = {p: b for p, b in download(backend, paths).items() if b}
    if not files:
        return
    reject_secrets([b.decode("utf-8", errors="replace") for b in files.values()])
    folder = ROOT / "development" / "checkpoints" / slugify(topic)
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "current_run_usage": usage}
    meta["source_receipts"] = get_source_receipts()
    merge_run_metrics(meta, previous_meta)
    meta["checkpoint_files"] = {path: f"artifact-{index}.txt" for index, path in enumerate(files)}
    for path, name in meta["checkpoint_files"].items():
        (folder / name).write_bytes(files[path])
    trace = [m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in messages]
    (folder / "lead-trace.json").write_text(redact(json.dumps(trace, ensure_ascii=False, indent=2)), encoding="utf-8")
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Draft checkpoint retained outside submission reports: " + str(folder), flush=True)


def reject_secrets(payloads):
    """The same gate protects published artifacts and ignored recovery checkpoints."""
    for payload in payloads:
        for name, value in os.environ.items():
            if len(value) >= 12 and any(w in name.upper() for w in ("API_KEY", "SECRET", "TOKEN", "PASSWORD")) and value in payload:
                raise RuntimeError("sandbox output contains a configured secret; nothing saved")


def normalize_source_ids(backend):
    """Normalize mechanical paper IDs and canonical heading case inside the sandbox."""
    code = f'''import json,re
from pathlib import Path
path=Path({SOURCES_PATH!r})
if path.exists():
    sources=json.loads(path.read_text())
    changed=False
    for s in sources:
        prefix={{'arxiv':'https://arxiv.org/abs/','hf-search':'https://huggingface.co/papers/','hf-daily':'https://huggingface.co/papers/'}}.get(s.get('source'))
        url=s.get('url','')
        if prefix and isinstance(url,str) and url.startswith(prefix) and s.get('id')==url:
            paper_id=url[len(prefix):]
            if re.fullmatch(r'(?:\\d{{4}}\\.\\d{{4,5}}|[A-Za-z.-]+/\\d{{7}})',paper_id):
                s['id']=paper_id
                changed=True
    if changed: path.write_text(json.dumps(sources,ensure_ascii=False,indent=2))
report_path=Path({REPORT_PATH!r})
if report_path.exists():
    text=report_path.read_text()
    for heading in ('TL;DR','Background','Trends and open problems','References'):
        text=re.sub(r'^##[ \\t]+'+re.escape(heading)+r'[ \\t]*$', '## '+heading,text,flags=re.M|re.I)
    report_path.write_text(text)
print('Source IDs normalized without changing provenance, URLs or dates')'''
    encoded = base64.b64encode(code.encode()).decode()
    result = backend.execute("python3 -c 'import base64; exec(base64.b64decode(\"" + encoded + "\"))'", timeout=120)
    if result.exit_code != 0:
        raise RuntimeError("source ID normalization failed: " + result.output[:1000])


def build_prompt(topic):
    """Keep the requested topic separate from system instructions, with an explicit date."""
    return (f"Research topic (user data): {json.dumps(topic, ensure_ascii=False)}\n"
            f"Research date: {datetime.now(timezone.utc).date().isoformat()}.\n"
            "Follow the complete research workflow in your system prompt. Research foundational work and "
            "developments within the last two years relative to this date. Finish only after the final sandbox "
            "report passes the citation validator, retains at least three source labels and has a citation audit.")


def _field(message, key, default=None):
    return message.get(key, default) if isinstance(message, dict) else getattr(message, key, default)


def summarize(messages, elapsed, model_name):
    """Count actual lead messages only; subagent token usage is not included."""
    calls, task_types = Counter(), Counter()
    input_tokens = output_tokens = 0
    for message in messages:
        for call in _field(message, "tool_calls", []) or []:
            name = call.get("name", "unknown")
            calls[name] += 1
            if name == "task":
                task_types[(call.get("args") or {}).get("subagent_type", "unknown")] += 1
        usage = _field(message, "usage_metadata", {}) or {}
        input_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": calls["task"],
            "tool_calls": dict(sorted(calls.items())), "tokens": {"input": input_tokens, "output": output_tokens},
            "tokens_scope": "lead_only; excludes subagent usage", "task_types": dict(task_types)}


def _validate_outputs(report, sources, meta, require_checker=True):
    problems = check(report, sources)
    if problems:
        raise RuntimeError("invalid citations: " + "; ".join(problems[:8]))
    metadata_problems = source_metadata_problems(sources, require_metadata=True)
    if metadata_problems:
        raise RuntimeError("invalid source records: " + "; ".join(metadata_problems[:8]))
    families = sorted({entry["source"] for entry in sources})
    if len(families) < 3:
        raise RuntimeError("final report needs at least 3 source families")
    if meta["subagent_calls"] < 3 or meta["task_types"].get("researcher", 0) < 3:
        raise RuntimeError("need at least 3 actual researcher delegations")
    if require_checker and not meta["task_types"].get("citation-checker", 0):
        raise RuntimeError("citation-checker was not delegated")
    structure_problems = report_structure_problems(report, require_template=True)
    if structure_problems:
        raise RuntimeError("invalid report structure: " + "; ".join(structure_problems))
    return families


def validate_audit(audit_bytes, sources, report_text=None, fetched_pages=None):
    """Require five supported samples matching the FINAL report and citation mapping."""
    try:
        audit = json.loads(audit_bytes.decode("utf-8")) if audit_bytes else None
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("citation-audit.json is not valid UTF-8 JSON") from exc
    if not isinstance(audit, list):
        raise RuntimeError("missing citation audit file" if audit is None else "citation audit must be a JSON array, got " + type(audit).__name__)
    sampled = audit_sample_numbers(sources)
    expected = {s["n"]: s["url"] for s in sources if s["n"] in sampled}
    seen, problems = set(), []
    for record in audit:
        if not isinstance(record, dict):
            problems.append("audit entries must be objects")
            continue
        n = record.get("n")
        if type(n) is not int or n not in expected:
            problems.append("audit has unknown/invalid source number")
            continue
        if n in seen:
            problems.append(f"audit has duplicate source [{n}]")
        seen.add(n)
        if record.get("url") != expected[n]:
            problems.append(f"audit [{n}] uses an outdated/wrong URL")
        if record.get("status") != "SUPPORTED":
            problems.append(f"audit [{n}] {record.get('status')}: {str(record.get('reason', ''))[:250]}")
        if not isinstance(record.get("reason"), str) or not record["reason"].strip():
            problems.append(f"audit [{n}] has no evidence reason")
        if report_text is not None:
            if record.get("report_sha256") != hashlib.sha256(report_text.encode("utf-8")).hexdigest():
                problems.append(f"audit [{n}] report hash missing/stale; audit the complete final report again")
            normal = lambda s: " ".join(s.split()).casefold()
            claim, evidence = record.get("claim"), record.get("evidence")
            body = report_text.split("## References", 1)[0]
            if not isinstance(claim, str) or f"[{n}]" not in claim or normal(claim) not in normal(body):
                problems.append(f"audit [{n}] claim must be an exact report-body excerpt containing [{n}]")
            page = (fetched_pages or {}).get(expected[n], "")
            if not isinstance(evidence, str) or len(evidence.strip()) < 12 or len(evidence.split()) > 10:
                problems.append(f"audit [{n}] evidence must be an exact short quote, <=10 words")
            elif not page or normal(evidence) not in normal(page):
                problems.append(f"audit [{n}] evidence not found in actual web_fetch text; fetch final URL and quote literally")
            else:
                record["evidence_sha256"] = hashlib.sha256(page.encode("utf-8")).hexdigest()
    if seen != expected.keys():
        problems.append(f"audit missing sources {sorted(expected.keys() - seen)}")
    if problems:
        raise RuntimeError("citation audit rejected report: " + "; ".join(problems[:12]))
    return audit


def validate_source_receipts(sources):
    """Bind final source metadata to records actually returned by host tools."""
    receipts = get_source_receipts()
    for source in sources:
        record = receipts.get(source["url"])
        if record is None or any(source.get(k) != record.get(k) for k in ("id", "title", "date", "source")):
            raise RuntimeError("source metadata must match real tool receipts; use save_retrieved_sources, never guess IDs")


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS, previous_meta=None, run_usage=None, require_source_receipts=False):
    """Validate everything before writing; preserve report/source bytes from sandbox exactly."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH, AUDIT_PATH])
    report_bytes, source_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip() or not source_bytes:
        raise RuntimeError("sandbox report/sources missing or empty; nothing saved")
    try:
        report = report_bytes.decode("utf-8")
        sources = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("sandbox output is not valid UTF-8/JSON; nothing saved") from exc
    meta = {"topic": topic, **summarize(messages, elapsed, model_name)}
    if run_usage is not None:
        meta["current_run_usage"] = run_usage
    merge_run_metrics(meta, previous_meta)
    families = _validate_outputs(report, sources, meta)
    if require_source_receipts:
        validate_source_receipts(sources)
    audit = validate_audit(files.get(AUDIT_PATH), sources, report, get_fetched_pages())
    # Detect accidental secret output before it can enter reports/ or Git.
    reject_secrets([report, source_bytes.decode("utf-8"), json.dumps(audit, ensure_ascii=False)])
    meta.update(n_sources=len(sources), source_families=families, citation_audit=audit,
                audit_version="quoted_evidence_v3_sampled", n_audited_sources=len(audit),
                audit_scope="5 source-claim samples; structural validator covers all citations")
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(topic)
    payloads = {f"{stem}.md": report_bytes, f"{stem}.sources.json": source_bytes,
                f"{stem}.meta.json": (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8")}
    previous = {name: (reports_dir / name).read_bytes() if (reports_dir / name).exists() else None for name in payloads}
    published = []
    # Stage all three files on the same filesystem; metadata is published last.
    with tempfile.TemporaryDirectory(prefix=".research-", dir=reports_dir) as staging:
        for name, payload in payloads.items():
            (Path(staging) / name).write_bytes(payload)
        try:
            for name in payloads:
                os.replace(Path(staging) / name, reports_dir / name)
                published.append(name)
        except OSError:
            for name in reversed(published):
                target = reports_dir / name
                if previous[name] is None:
                    target.unlink(missing_ok=True)
                else:
                    target.write_bytes(previous[name])
            raise
    return reports_dir / f"{stem}.md"


def main(topic, *, repair_existing=False, guidance="", fresh=False, resume_checkpoint=False, enforce_source_receipts=False, audit_only=False):
    """Return 0 on success, 1 on failure, 2 for missing topic. Always clean up sandbox."""
    if not topic.strip():
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    start = time.monotonic()
    clear_fetched_pages()
    budget = None
    try:
        budget = RunTokenBudget(int(os.getenv("LAB_MAX_TOTAL_TOKENS", "250000")))
        configured = os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL") or ""
        if "<" in configured or ">" in configured:
            raise RuntimeError("LAB_MODEL is still a placeholder; set a real tool-calling model in .env")
        model = make_model()
        if hasattr(model, "max_tokens"):
            model = model.model_copy(update={"max_tokens": 8192})
        model_name = configured or getattr(model, "model_name", "unknown")
        previous_meta = None
        seeds = {}
        if repair_existing and not resume_checkpoint:
            stem = REPORTS / slugify(topic)
            previous_meta = json.loads(stem.with_suffix(".meta.json").read_text(encoding="utf-8"))
            seeds = {REPORT_PATH: stem.with_suffix(".md").read_bytes(),
                     SOURCES_PATH: stem.with_suffix(".sources.json").read_bytes()}
        else:
            folder = ROOT / "development" / "checkpoints" / slugify(topic)
            if not fresh and (folder / "meta.json").exists() and (resume_checkpoint or not (REPORTS / (slugify(topic) + ".md")).exists()):
                previous_meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
                if previous_meta["topic"] != topic:
                    raise RuntimeError("checkpoint belongs to a different topic")
                seeds = {}
                for path, name in previous_meta["checkpoint_files"].items():
                    if path not in {REPORT_PATH, SOURCES_PATH, AUDIT_PATH} and not (path.startswith(NOTES_DIR + "/") and path.endswith(".md")):
                        raise RuntimeError("invalid checkpoint sandbox path")
                    if Path(name).name != name:
                        raise RuntimeError("invalid checkpoint filename")
                    seeds[path] = (folder / name).read_bytes()
                reject_secrets([b.decode("utf-8", errors="replace") for b in seeds.values()])
                restore_source_receipts(previous_meta.get("source_receipts", {}))
                print("Resuming draft checkpoint; completed research will be reused.", flush=True)
        review_mode = repair_existing or (REPORT_PATH in seeds and SOURCES_PATH in seeds)
        if audit_only and not (REPORT_PATH in seeds and SOURCES_PATH in seeds):
            raise RuntimeError("audit_only requires an existing report and sources")
        receipt_mode = not review_mode or enforce_source_receipts
        print(f"Starting survey: {topic} | model: {model_name}", flush=True)
        with open_sandbox() as backend:
            prepared = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if prepared.exit_code != 0:
                raise RuntimeError("cannot create sandbox workspace")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                             FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(), **seeds})
            if SOURCES_PATH in seeds:
                normalize_source_ids(backend)
            print("Sandbox ready; lead is planning, researching and validating.", flush=True)
            options = {"repair_existing": review_mode, "source_receipts": receipt_mode}
            agent = build_lead_agent(backend, model, stage="draft", **options)
            initial = "DRAFT_PHASE. " + (f"Review existing survey for topic: {topic}. Resolve every claim against its final source. "
                       f"Independent reviewer findings: {guidance}" if review_mode else build_prompt(topic) + "\n" + guidance)
            if seeds and not review_mode:
                initial += "\nCompleted researcher notes are already uploaded: " + ", ".join(seeds) + ". Reuse those actual notes; do not repeat completed research. Original delegations are recorded in previous metadata. Continue synthesis, finalization, validation and audit."
            state = {"messages": [{"role": "user", "content": initial}]}
            config = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": uuid.uuid4().hex},
                      "callbacks": [budget]}
            completed_audits = []
            phase_prefix = []
            def invoke_phase(phase_state):
                try:
                    return agent.invoke(phase_state, config=config)
                except Exception:
                    try:
                        messages = phase_prefix + agent.get_state(config).values.get("messages", [])
                        checkpoint_draft(backend, topic, messages, time.monotonic() - start, model_name, budget.snapshot(), previous_meta)
                    except Exception as checkpoint_error:
                        print("Could not preserve draft: " + redact(checkpoint_error), flush=True)
                    raise
            for round_index in range(MAX_VALIDATION_ROUNDS):
                phase_prefix = completed_audits
                result = {"messages": []} if audit_only and round_index == 0 else invoke_phase(state)
                result["messages"] = result.get("messages", []) + completed_audits
                try:
                    # Restore trusted code before independently checking in the sandbox.
                    upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
                    normalize_source_ids(backend)
                    finalized = backend.execute(f"python3 {FINALIZER_PATH}", timeout=120)
                    if finalized.exit_code != 0:
                        raise RuntimeError("sandbox finalization failed: " + finalized.output[:1500])
                    validation = backend.execute(f"python3 {VALIDATOR_PATH}")
                    if validation.exit_code != 0 or not validation.output.startswith("OK:"):
                        raise RuntimeError("sandbox citation validation failed: " + validation.output[:1500])
                    print(validation.output.strip(), flush=True)
                    draft_files = download(backend, [REPORT_PATH, SOURCES_PATH])
                    draft_meta = {"topic": topic, **summarize(result.get("messages", []), time.monotonic() - start, model_name)}
                    merge_run_metrics(draft_meta, previous_meta)
                    _validate_outputs(draft_files[REPORT_PATH].decode("utf-8"),
                                      json.loads(draft_files[SOURCES_PATH].decode("utf-8")), draft_meta, require_checker=False)
                    if receipt_mode:
                        validate_source_receipts(json.loads(draft_files[SOURCES_PATH]))
                    draft_messages = result.get("messages", [])
                    phase_prefix = draft_messages
                    draft_agent = agent
                    agent = build_lead_agent(backend, model, stage="audit", **options)
                    result = invoke_phase({"messages": [{"role": "user", "content":
                        "AUDIT_PHASE. Host deterministic finalization and validation passed. Do not modify the report "
                        "or sources. Delegate citation-checker once to check the five claims selected by audit_prepare "
                        "from the CURRENT final files. Use audit_record, actual web_fetch evidence and final mapping; "
                        "ignore old numbers. Return the audit path and specific unsupported claims, without editing."}]})
                    audit_messages = result.get("messages", [])
                    completed_audits += audit_messages
                    result["messages"] = draft_messages + audit_messages
                    agent = draft_agent
                    phase_prefix = completed_audits
                    kwargs = {"previous_meta": previous_meta} if previous_meta is not None else {}
                    output = save_outputs(backend, topic, result.get("messages", []), time.monotonic() - start, model_name,
                                          run_usage=budget.snapshot(), require_source_receipts=receipt_mode, **kwargs)
                    break
                except RuntimeError as exc:
                    if round_index + 1 == MAX_VALIDATION_ROUNDS:
                        checkpoint_draft(backend, topic, result.get("messages", []), time.monotonic() - start,
                                         model_name, budget.snapshot(), previous_meta)
                        raise
                    feedback = redact(exc)
                    print(f"Validation requires one sandbox repair: {feedback}", flush=True)
                    # The same checkpoint/thread preserves history and PRIVATE middleware counters.
                    state = {"messages": [{"role": "user", "content":
                        "DRAFT_PHASE. Final validation rejected the artifacts: " + feedback +
                        ". Repair only from retrieved evidence INSIDE the sandbox. Do not invent or relabel sources. "
                        "Run finalize_citations.py after changes, then the validator. Ensure three source families "
                        "remain, all required sections exist, and rerun the full citation audit using the FINAL "
                        "source mapping. The controller will ask AUDIT_PHASE after your draft is standardized. "
                        "Fix the reported claims and return draft paths; do not start another audit yet."}]}
        print(f"Saved: {output}", flush=True)
        if budget is not None:
            print("Measured run usage: " + json.dumps(budget.snapshot()), flush=True)
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {redact(exc)}", file=sys.stderr, flush=True)
        if budget is not None:
            print("Measured run usage: " + json.dumps(budget.snapshot()), flush=True)
        return 1


if __name__ == "__main__":
    repair = len(sys.argv) > 1 and sys.argv[1] == "--repair-existing"
    sys.exit(main(" ".join(sys.argv[2:] if repair else sys.argv[1:]), repair_existing=repair))
