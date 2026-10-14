"""Focused deterministic tests. Fixtures are synthetic and never submission reports."""
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

import agents
import research
import tools
from check_citations import check


def fixture():
    sources = [
        {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "A", "date": "2025-01-01", "source": "arxiv"},
        {"n": 2, "id": "2501.00002", "url": "https://huggingface.co/papers/2501.00002", "title": "B", "date": "2025-01-02", "source": "hf-search"},
        {"n": 3, "id": "page", "url": "https://example.org/page", "title": "C", "date": "", "source": "web"},
    ]
    body = "# Test\n## TL;DR\n- Result [1].\n## Background\nFoundation [2].\n"
    body += "## Theme A\nA [1].\n## Theme B\nB [2].\n## Theme C\nC [3].\n## Trends and open problems\nGap [3].\n"
    report = body + "## References\n" + "\n".join(f"[{s['n']}] {s['title']}. {s['url']} ({s['date']})" for s in sources)
    messages = [AIMessage(content="", tool_calls=[{"id": str(i), "name": "task", "args": {"subagent_type": name}}
                for i, name in enumerate(["researcher"] * 3 + ["citation-checker"])],
                usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25})]
    return report, sources, messages


def audit_bytes(sources):
    claims = {1: "Result [1].", 2: "Foundation [2].", 3: "C [3]."}
    return json.dumps([{"n": s["n"], "url": s["url"], "status": "SUPPORTED", "reason": "Synthetic test evidence",
                        "claim": claims[s["n"]], "evidence": "Synthetic test evidence",
                        "report_sha256": research.hashlib.sha256(fixture()[0].encode()).hexdigest()}
                       for s in sources]).encode()


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.report, self.sources, _ = fixture()

    def test_valid_report_and_grouped_citations(self):
        self.assertEqual(check(self.report, self.sources), [])
        for group in ("[1, 2, 3]", "[1-3]", "[1–3]"):
            refs = self.report.split("## References")[1]
            self.assertEqual(check(group + "\n## References" + refs, self.sources), [])

    def test_reference_list_cannot_count_as_body_citations(self):
        report = "No claims.\n## References" + self.report.split("## References")[1]
        self.assertTrue(any("never cited" in p for p in check(report, self.sources)))

    def test_code_and_markdown_links_do_not_count(self):
        for text in ("```python\n[1][2][3]\n```", "~~~\n[1][2][3]\n~~~", "`[1][2][3]`", "[1](https://a.org) [2](https://b.org) [3](https://c.org)"):
            report = text + "\n## References" + self.report.split("## References")[1]
            self.assertEqual(sum("never cited" in p for p in check(report, self.sources)), 3)

    def test_fake_heading_in_code_and_duplicate_real_heading(self):
        self.assertTrue(check("```\n" + self.report + "\n```", self.sources))
        self.assertEqual(check(self.report + "\n```\n## References\n```", self.sources), [])
        self.assertTrue(check(self.report + "\n## References\n", self.sources))

    def test_unknown_missing_duplicate_and_multiple_urls(self):
        variants = [self.report.replace("Result [1]", "Result [9]"),
                    self.report.replace("[2] B. https://huggingface.co/papers/2501.00002 (2025-01-02)", ""),
                    self.report + "\n[1] A. https://arxiv.org/abs/2501.00001",
                    self.report.replace("[1] A.", "[1] A. https://wrong.org"),
                    self.report.replace("[3] C. https://example.org/page", "[3] C. https://wrong.org")]
        for report in variants:
            self.assertTrue(check(report, self.sources))

    def test_malformed_sources_are_errors_not_crashes(self):
        for sources in ([], {}, [None], [{"n": True, "url": "bad"}], [{"n": [], "url": []}],
                        [{"n": 1, "url": "https://example.org", "source": []}],
                        [self.sources[0], self.sources[0]]):
            self.assertTrue(check(self.report, sources))

    def test_malformed_citation_groups_are_rejected_without_crashing(self):
        for group in ("[1-2-3]", "[3-1]", "[1,,2]", "[1-99999999]", "[" + "9" * 5000 + "]"):
            problems = check(self.report.replace("Result [1]", "Result " + group), self.sources)
            self.assertIn("invalid citation group or range", problems)

    def test_parenthesised_prose_is_not_a_markdown_link(self):
        refs = self.report.split("## References")[1]
        self.assertEqual(check("Evidence [1] (2025), [2] (related), [3].\n## References" + refs, self.sources), [])

    def test_consistent_citations_still_reject_mislabelled_source_urls(self):
        sources = [dict(s) for s in self.sources]
        sources[1]["id"] = "Project2026"
        sources[1]["url"] = "https://github.com/example/project"
        report = self.report.replace(self.sources[1]["url"], sources[1]["url"])
        self.assertTrue(any("hf-search" in p and "URL" in p for p in check(report, sources)))

    def test_source_schema_reports_all_errors_together(self):
        sources = [dict(s) for s in self.sources]
        sources[0]["id"] += "v2"
        sources[1]["id"] = "wrong"
        problems = check(self.report, sources)
        self.assertTrue(any("arxiv" in p for p in problems))
        self.assertTrue(any("hf-search" in p for p in problems))

    def test_sandbox_validator_checks_report_template_before_ok(self):
        wrong_case = self.report.replace("## Trends and open problems", "## Trends and Open Problems")
        self.assertTrue(any("Trends and open problems" in p for p in check(wrong_case, self.sources)))
        too_many = self.report.replace("## References", "## Extra 1\n## Extra 2\n## Extra 3\n## Extra 4\n## References")
        self.assertTrue(any("3-6" in p for p in check(too_many, self.sources)))

    def test_bibliography_in_body_cannot_be_an_audit_claim(self):
        wrong = self.report.replace("## References", "[1] A. https://arxiv.org/abs/2501.00001\n\n## References")
        self.assertTrue(any("bibliography entry outside" in p for p in check(wrong, self.sources)))

    def test_duplicate_case_variant_and_uncited_survey_paragraph_are_errors(self):
        variant = self.report.replace("## References", "## Trends and Open Problems\nAnother result [1].\n\n## References")
        self.assertTrue(any("non-canonical heading" in p for p in check(variant, self.sources)))
        missing = self.report.replace("## References", "\nThis is a substantive scientific paragraph that makes several claims without providing any evidence citation.\n\n## References")
        self.assertTrue(any("paragraph needs" in p for p in check(missing, self.sources)))

    def test_corrupted_tail_after_citation_is_rejected(self):
        corrupted = self.report.replace("Result [1].", "Result [1].ent, duplicated sentence fragment.")
        self.assertTrue(any("corrupted" in p for p in check(corrupted, self.sources)))

    def test_each_survey_bullet_needs_its_own_citation(self):
        report = self.report.replace("- Result [1].", "- Result [1].\n- Another substantive finding without supporting citations.")
        self.assertTrue(any("bullet needs" in p for p in check(report, self.sources)))


class ToolTests(unittest.TestCase):
    @patch("tools.time.sleep")
    def test_id_lookup_records_actual_metadata_and_first_provenance(self, sleep):
        tools.clear_fetched_pages()
        xml = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.00001v2</id><title>Actual title</title><summary>Actual abstract</summary><published>2025-01-02T00:00:00Z</published></entry></feed>'
        with patch("tools._request", return_value=httpx.Response(200, text=xml)) as request:
            tools.arxiv_search.invoke({"query": "2501.00001"})
            self.assertEqual(request.call_args.kwargs["params"], {"id_list": "2501.00001"})
        url = "https://arxiv.org/abs/2501.00001"
        with patch("tools._exa_call", return_value="# Different page heading\nPublished: 2099-01-01\nContent"):
            tools.web_fetch.invoke({"url": url})
        self.assertEqual(tools.get_source_receipts()[url], {"id": "2501.00001", "url": url,
                         "title": "Actual title", "date": "2025-01-02", "source": "arxiv"})
        paper = {"id": "2501.00002", "title": "Actual HF title", "publishedAt": "2025-01-03T00:00:00Z"}
        with patch("tools._request", return_value=httpx.Response(200, json=paper)) as request:
            tools.hf_search_papers.invoke({"query": "2501.00002"})
            self.assertEqual(request.call_args.args[1], "https://huggingface.co/api/papers/2501.00002")
        receipt = tools.get_source_receipts()["https://huggingface.co/papers/2501.00002"]
        self.assertEqual((receipt["title"], receipt["date"], receipt["source"]), ("Actual HF title", "2025-01-03", "hf-search"))
        tools.clear_fetched_pages()
        self.assertEqual(tools.get_source_receipts(), {})

    def test_failed_fetch_and_new_run_cannot_reuse_old_evidence(self):
        url = "https://example.org/evidence"
        tools.clear_fetched_pages()
        with patch("tools._exa_call", side_effect=["Retrieved evidence", "NO RESULTS", "Fresh evidence", RuntimeError("offline")]):
            tools.web_fetch.invoke({"url": url})
            self.assertIn(url, tools.get_fetched_pages())
            tools.web_fetch.invoke({"url": url})
            self.assertNotIn(url, tools.get_fetched_pages())
            tools.web_fetch.invoke({"url": url})
            tools.web_fetch.invoke({"url": url})
            self.assertNotIn(url, tools.get_fetched_pages())
        with patch("tools._exa_call", return_value="Retrieved evidence"):
            tools.web_fetch.invoke({"url": url})
        tools.clear_fetched_pages()
        self.assertEqual(tools.get_fetched_pages(), {})

    @patch("tools.time.sleep")
    def test_retry_server_delay_and_no_final_sleep(self, sleep):
        calls = []
        def fail():
            calls.append(1)
            raise tools.RetryableError("temporary", retry_after=8)
        with self.assertRaises(tools.RetryableError):
            tools.with_retry(fail, attempts=3, cap=5)
        self.assertEqual(len(calls), 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [5, 5])

    @patch("tools.time.sleep")
    @patch("tools.random.uniform", return_value=0.5)
    def test_exponential_jitter_cap_and_recovery(self, jitter, sleep):
        remaining = [1, 2, 3, 4]
        def call():
            if remaining:
                remaining.pop()
                raise tools.RetryableError("retry")
            return "ok"
        self.assertEqual(tools.with_retry(call, cap=4), "ok")
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.5, 2.5, 4, 4])

    @patch("tools.time.sleep")
    def test_permanent_error_never_retried(self, sleep):
        with self.assertRaises(ValueError):
            tools.with_retry(lambda: int("bad"))
        sleep.assert_not_called()

    def test_transient_http_and_transport_mapping(self):
        for status in (429, 500, 502, 503, 504):
            response = httpx.Response(status, headers={"Retry-After": "2"}, request=httpx.Request("GET", "https://example.org"))
            with patch("tools.httpx.request", return_value=response):
                with self.assertRaises(tools.RetryableError) as ctx:
                    tools._request("GET", "https://example.org")
                self.assertEqual(ctx.exception.retry_after, 2)
        with patch("tools.httpx.request", side_effect=httpx.ConnectError("offline")):
            with self.assertRaises(tools.RetryableError):
                tools._request("GET", "https://example.org")
        response = httpx.Response(401, request=httpx.Request("GET", "https://example.org"))
        with patch("tools.httpx.request", return_value=response):
            with self.assertRaises(httpx.HTTPStatusError):
                tools._request("GET", "https://example.org")

    @patch("tools.time.sleep")
    def test_arxiv_normalises_and_throttles_each_attempt(self, sleep):
        xml = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.00001v2</id><title>A\n B</title><summary>Text</summary><published>2025-01-01T00:00:00Z</published></entry></feed>'
        with patch("tools._ARXIV_LAST_CALL", None), patch("tools.time.monotonic", side_effect=[10, 11, 13]), patch("tools._request", return_value=httpx.Response(200, text=xml)) as request:
            first = json.loads(tools.arxiv_search.invoke({"query": 'all:"world" AND model'}))
            tools.arxiv_search.invoke({"query": "world model"})
            self.assertEqual(first[0]["url"], "https://arxiv.org/abs/2501.00001")
            self.assertEqual(first[0]["title"], "A B")
            self.assertEqual(request.call_args.kwargs["params"]["search_query"], "all:world AND all:model")
            self.assertIn(2, [c.args[0] for c in sleep.call_args_list])
        with patch("tools._request") as request:
            self.assertEqual(tools.arxiv_search.invoke({"query": " : AND OR !!"}), "NO RESULTS")
            request.assert_not_called()

    def test_hf_fields_filter_sort_skip_and_empty(self):
        items = [{"paper": {"id": "a", "title": "World model", "ai_summary": "Short", "upvotes": 2}},
                 {"paper": {"id": "b", "title": "WORLD MODEL", "upvotes": 8}}, {"paper": {"title": "Missing id"}}]
        with patch("tools._request", return_value=httpx.Response(200, json=items)):
            records = json.loads(tools.hf_daily_papers.invoke({"keyword": "world model"}))
            self.assertEqual([r["id"] for r in records], ["b", "a"])
            self.assertEqual(records[1]["summary"], "Short")
            self.assertEqual(tools.hf_daily_papers.invoke({"keyword": "unrelated"}), "NO RESULTS")
            self.assertEqual(len(json.loads(tools.hf_search_papers.invoke({"query": "world"}))), 2)

    @patch("tools.time.sleep")
    def test_exa_sse_rate_limit_retry_and_array_fetch(self, sleep):
        limited = {"jsonrpc": "2.0", "id": 1, "result": {"_meta": {"rateLimited": True}, "content": [{"type": "text", "text": "try later"}]}}
        ok = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "evidence"}]}}
        responses = [httpx.Response(200, text="event: message\ndata: " + json.dumps(p) + "\n\n") for p in (limited, ok)]
        with patch("tools._request", side_effect=responses) as request:
            self.assertEqual(tools.web_fetch.invoke({"url": "https://example.org"}), "evidence")
            self.assertEqual(request.call_args.kwargs["json"]["params"]["arguments"], {"urls": ["https://example.org"]})
        self.assertEqual(sleep.call_count, 1)

    def test_errors_are_strings_and_secrets_are_redacted(self):
        sample = "synthetic-test-value"
        with patch.dict(os.environ, {"EXA_API_KEY": sample}), patch("tools._request", side_effect=ValueError("https://mcp.exa.ai?exaApiKey=" + sample)):
            for fn, args in ((tools.hf_search_papers, {"query": "a"}), (tools.web_search, {"query": "a"}), (tools.web_fetch, {"url": "https://example.org"})):
                result = fn.invoke(args)
                self.assertTrue(result.startswith("ERROR:"))
                self.assertNotIn(sample, result)

    def test_real_exa_content_and_positive_quota_are_not_throttling(self):
        self.assertFalse(tools._rate_limited({"rateLimitRemaining": 100}, "avoid rate-limited requests"))
        self.assertFalse(tools._rate_limited({"rateLimit": {"limit": 100, "remaining": 50}}, "# Rate limiting\nA paper"))
        self.assertTrue(tools._rate_limited({"error": {"code": "RATE_LIMIT_EXCEEDED"}}, ""))
        self.assertTrue(tools._rate_limited({}, "You've hit Exa's free MCP rate limit. Try later."))


class ResearchTests(unittest.TestCase):
    def test_failed_draft_is_retained_outside_reports_with_measured_phase_totals(self):
        report, sources, messages = fixture()
        previous = {"topic": "topic", **research.summarize(messages, 2, "original")}
        files = {research.REPORT_PATH: report.encode(), research.SOURCES_PATH: json.dumps(sources).encode()}
        backend = MagicMock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="[]")
        with tempfile.TemporaryDirectory() as temp, patch("research.ROOT", Path(temp)), patch("research.download", return_value=files), redirect_stdout(StringIO()):
            research.checkpoint_draft(backend, "topic", messages, 3, "mini", {"input": 50}, previous)
            folder = Path(temp) / "development/checkpoints/topic"
            meta = json.loads((folder / "meta.json").read_text())
            self.assertEqual((folder / meta["checkpoint_files"][research.REPORT_PATH]).read_bytes(), report.encode())
            self.assertEqual(meta["subagent_calls"], 8)
            self.assertEqual(meta["elapsed_s"], 5)
            self.assertEqual(meta["phases"][-1]["current_run_usage"], {"input": 50})
            self.assertFalse((Path(temp) / "reports").exists())

    def test_shared_budget_reserves_parallel_calls_and_counts_actual_usage(self):
        from langchain_core.messages import HumanMessage
        from langchain_core.outputs import LLMResult
        budget = research.RunTokenBudget(18000)
        messages = [[HumanMessage(content="short request")]]
        budget.on_chat_model_start({}, messages, run_id="a")
        budget.on_chat_model_start({}, messages, run_id="b")
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            budget.on_chat_model_start({}, messages, run_id="c")
        budget.on_llm_end(LLMResult(generations=[], llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 25}}), run_id="a")
        budget.on_llm_error(RuntimeError("offline"), run_id="b")
        self.assertEqual(budget.snapshot()["input"], 100)
        self.assertEqual(budget.snapshot()["output"], 25)
        self.assertEqual(budget.pending, {})
        budget.on_chat_model_start({}, messages, run_id="c")

    def test_missing_usage_fails_closed_and_checkpoint_rejects_secrets(self):
        from langchain_core.messages import HumanMessage
        from langchain_core.outputs import LLMResult
        budget = research.RunTokenBudget()
        with self.assertRaisesRegex(RuntimeError, "omitted token usage"):
            budget.on_llm_end(LLMResult(generations=[]), run_id="a")
        with self.assertRaisesRegex(RuntimeError, "omitted token usage"):
            budget.on_chat_model_start({}, [[HumanMessage(content="request")]], run_id="b")
        self.assertEqual(budget.snapshot()["unmetered_calls"], 1)
        backend = MagicMock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="[]")
        secret = "synthetic-private-secret-value"
        with tempfile.TemporaryDirectory() as temp, patch("research.ROOT", Path(temp)), \
             patch.dict(os.environ, {"TEST_API_KEY": secret}), \
             patch("research.download", return_value={research.REPORT_PATH: secret.encode()}):
            with self.assertRaisesRegex(RuntimeError, "configured secret"):
                research.checkpoint_draft(backend, "topic", [], 0, "test", {})
            self.assertFalse((Path(temp) / "development").exists())

    def setUp(self):
        _, sources, _ = fixture()
        self.pages = {s["url"]: "Synthetic test evidence from a deterministic fixture." for s in sources}
        patcher = patch("research.get_fetched_pages", return_value=self.pages)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_slug_cannot_escape_and_is_bounded(self):
        for value in ("../../x", "", "...", "a" * 90, "中文", "CON"):
            slug = research.slugify(value)
            self.assertTrue(slug)
            self.assertLessEqual(len(slug), 60)
            self.assertNotIn("/", slug)
            self.assertNotIn("\\", slug)

    def test_real_message_usage_counts_and_dict_messages(self):
        _, _, messages = fixture()
        result = research.summarize(messages, 1.234, "test")
        self.assertEqual(result["tokens"], {"input": 20, "output": 5})
        self.assertEqual(result["subagent_calls"], 4)
        self.assertEqual(result["task_types"]["researcher"], 3)
        self.assertEqual(research.summarize([{"tool_calls": [{"name": "execute"}]}], 0, "test")["tool_calls"], {"execute": 1})

    def test_invalid_downloads_write_nothing(self):
        _, _, messages = fixture()
        for files in ({}, {research.REPORT_PATH: b"", research.SOURCES_PATH: b"[]"},
                      {research.REPORT_PATH: b"abc", research.SOURCES_PATH: b"not JSON"}):
            with tempfile.TemporaryDirectory() as temp, patch("research.download", return_value=files):
                with self.assertRaises(RuntimeError):
                    research.save_outputs(None, "topic", messages, 1, "test", Path(temp))
                self.assertEqual(list(Path(temp).iterdir()), [])

    def test_valid_downloads_preserve_exact_bytes(self):
        report, sources, messages = fixture()
        source_bytes = json.dumps(sources, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as temp, patch("research.download", return_value={research.REPORT_PATH: report.encode(), research.SOURCES_PATH: source_bytes, research.AUDIT_PATH: audit_bytes(sources)}):
            path = research.save_outputs(None, "test topic", messages, 1, "test", Path(temp))
            self.assertEqual(path.read_bytes(), report.encode())
            self.assertEqual((Path(temp) / "test-topic.sources.json").read_bytes(), source_bytes)
            self.assertEqual(len(list(Path(temp).iterdir())), 3)

    def test_final_metadata_changes_after_preflight_are_rejected(self):
        report, sources, messages = fixture()
        receipts = {s["url"]: dict(s) for s in sources}
        with patch("research.get_source_receipts", return_value=receipts):
            research.validate_source_receipts(sources)
            for field, value in (("title", "invented title"), ("date", "2099-01-01"), ("source", "hf-daily")):
                changed = [dict(s) for s in sources]
                changed[0][field] = value
                files = {research.REPORT_PATH: report.encode(), research.SOURCES_PATH: json.dumps(changed).encode(),
                         research.AUDIT_PATH: audit_bytes(sources)}
                with tempfile.TemporaryDirectory() as temp, patch("research.download", return_value=files), \
                     patch("research._validate_outputs", return_value=["arxiv", "hf-search", "web"]):
                    with self.assertRaisesRegex(RuntimeError, "real tool receipts"):
                        research.save_outputs(None, "topic", messages, 1, "test", Path(temp), require_source_receipts=True)
                    self.assertEqual(list(Path(temp).iterdir()), [])

    def test_partial_publish_rolls_back_existing_files(self):
        report, sources, messages = fixture()
        with tempfile.TemporaryDirectory() as temp, patch("research.download", return_value={research.REPORT_PATH: report.encode(), research.SOURCES_PATH: json.dumps(sources).encode(), research.AUDIT_PATH: audit_bytes(sources)}):
            target = Path(temp) / "topic.md"
            target.write_bytes(b"previous run")
            original = research.os.replace
            calls = []
            def replace(src, dst):
                calls.append(1)
                if len(calls) == 2:
                    raise OSError("synthetic disk failure")
                original(src, dst)
            with patch("research.os.replace", side_effect=replace), self.assertRaises(OSError):
                research.save_outputs(None, "topic", messages, 1, "test", Path(temp))
            self.assertEqual(target.read_bytes(), b"previous run")
            self.assertEqual(len(list(Path(temp).iterdir())), 1)

    def test_provenance_and_delegation_gates(self):
        report, sources, messages = fixture()
        sources[0]["source"] = "hf-daily"
        with self.assertRaises(RuntimeError):
            research._validate_outputs(report, sources, research.summarize(messages, 1, "test"))
        report, sources, _ = fixture()
        with self.assertRaises(RuntimeError):
            research._validate_outputs(report, sources, research.summarize([], 1, "test"))

    def test_main_repairs_source_mismatch_in_same_sandbox_with_real_output_gates(self):
        report, sources, messages = fixture()
        wrong = [dict(s) for s in sources]
        wrong[0]["id"] += "v2"
        good_files = {research.REPORT_PATH: report.encode(), research.SOURCES_PATH: json.dumps(sources).encode(), research.AUDIT_PATH: audit_bytes(sources)}
        bad_files = {research.REPORT_PATH: report.encode(), research.SOURCES_PATH: json.dumps(wrong).encode(), research.AUDIT_PATH: audit_bytes(sources)}
        backend, agent = MagicMock(), MagicMock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="OK: 3 sources, all citations resolve")
        state = {"messages": messages}
        agent.invoke.return_value = state
        save_original = research.save_outputs
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"LAB_MODEL": "test"}), \
             patch("research.make_model", return_value=SimpleNamespace(model_name="test")), \
             patch("research.open_sandbox") as context, patch("research.upload"), \
             patch("research.build_lead_agent", return_value=agent), \
             patch("research.get_source_receipts", return_value={s["url"]: s for s in sources}), \
             patch("research.download", side_effect=[bad_files, good_files, good_files]), \
             patch("research.save_outputs", wraps=lambda *args, **kwargs: save_original(*args, reports_dir=Path(temp), **kwargs)), \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            context.return_value.__enter__.return_value = backend
            self.assertEqual(research.main("topic"), 0)
            self.assertEqual(agent.invoke.call_count, 3)
            repair_state = agent.invoke.call_args_list[1].args[0]
            self.assertEqual(len(repair_state["messages"]), 1)
            first_config = agent.invoke.call_args_list[0].kwargs["config"]
            self.assertTrue(all(first_config == call.kwargs["config"] for call in agent.invoke.call_args_list))
            self.assertTrue(first_config["configurable"]["thread_id"])
            self.assertIn("arxiv", repair_state["messages"][-1]["content"])
            self.assertIn("id/URL mismatch", repair_state["messages"][-1]["content"])
            self.assertIn("INSIDE the sandbox", repair_state["messages"][-1]["content"])
            self.assertEqual(json.loads((Path(temp) / "topic.sources.json").read_bytes()), sources)
            context.return_value.__exit__.assert_called_once()

    def test_real_graph_checkpoint_preserves_private_model_limit_across_invokes(self):
        from langchain.agents import create_agent
        from langchain.agents.middleware import ModelCallLimitMiddleware
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langgraph.checkpoint.memory import InMemorySaver
        graph = create_agent(FakeListChatModel(responses=["first", "second"]), checkpointer=InMemorySaver(),
                             middleware=[ModelCallLimitMiddleware(run_limit=5, thread_limit=1, exit_behavior="end")])
        config = {"configurable": {"thread_id": "synthetic-limit-test"}}
        first = graph.invoke({"messages": [{"role": "user", "content": "first request"}]}, config=config)
        self.assertEqual(first["messages"][-1].content, "first")
        second = graph.invoke({"messages": [{"role": "user", "content": "repair request"}]}, config=config)
        self.assertNotEqual(second["messages"][-1].content, "second")
        self.assertIn("limit", second["messages"][-1].content.lower())
        self.assertEqual(sum(m.content == "first" for m in second["messages"]), 1)

    def test_audit_must_cover_final_mapping_and_support_all_claims(self):
        _, sources, _ = fixture()
        self.assertEqual(len(research.validate_audit(audit_bytes(sources), sources)), 3)
        for audit in (None, [], [{"n": 1, "url": sources[0]["url"], "status": "SUPPORTED", "reason": "test"}]):
            with self.assertRaises(RuntimeError):
                research.validate_audit(json.dumps(audit).encode(), sources)
        audit = json.loads(audit_bytes(sources))
        audit[1]["url"] = "https://example.org/outdated"
        with self.assertRaises(RuntimeError):
            research.validate_audit(json.dumps(audit).encode(), sources)

    def test_audit_proof_must_match_real_report_and_actual_fetched_text(self):
        report, sources, _ = fixture()
        audit = research.validate_audit(audit_bytes(sources), sources, report, self.pages)
        self.assertTrue(all(len(a["evidence_sha256"]) == 64 for a in audit))
        with self.assertRaisesRegex(RuntimeError, "hash missing/stale"):
            research.validate_audit(audit_bytes(sources), sources, report + "\nChanged claim [1].", self.pages)
        for key, wrong in (("claim", "Invented benchmark [1]."), ("evidence", "A paraphrase that was never retrieved")):
            bad = json.loads(audit_bytes(sources))
            bad[0][key] = wrong
            with self.assertRaises(RuntimeError):
                research.validate_audit(json.dumps(bad).encode(), sources, report, self.pages)
        with self.assertRaises(RuntimeError):
            research.validate_audit(audit_bytes(sources), sources, report, {})

    def test_followup_metadata_sums_only_measured_phases(self):
        report, sources, messages = fixture()
        previous = {"topic": "topic", **research.summarize(messages, 10, "old-model")}
        files = {research.REPORT_PATH: report.encode(), research.SOURCES_PATH: json.dumps(sources).encode(),
                 research.AUDIT_PATH: audit_bytes(sources)}
        with tempfile.TemporaryDirectory() as temp, patch("research.download", return_value=files):
            research.save_outputs(None, "topic", messages, 2, "new-model", Path(temp), previous_meta=previous)
            meta = json.loads((Path(temp) / "topic.meta.json").read_bytes())
            self.assertEqual(meta["subagent_calls"], 8)
            self.assertEqual(meta["tokens"], {"input": 40, "output": 10})
            self.assertEqual(meta["elapsed_s"], 12)
            self.assertEqual([p["model"] for p in meta["phases"]], ["old-model", "new-model"])
        audit = json.loads(audit_bytes(sources))
        audit[1]["status"] = "UNSUPPORTED"
        with self.assertRaises(RuntimeError):
            research.validate_audit(json.dumps(audit).encode(), sources)

    def test_graph_compiles_with_only_bounded_specialists(self):
        from deepagents.backends import StateBackend
        graph = agents.build_lead_agent(StateBackend(), ChatOpenAI(model="gpt-4.1-mini", api_key="test"))
        tools_node = graph.nodes["tools"].bound
        task = tools_node.tools_by_name["task"]
        description = task.description
        self.assertIn("researcher", description)
        self.assertIn("citation-checker", description)
        self.assertNotIn("\n- general-purpose:", description)
        self.assertIn("write_todos", tools_node.tools_by_name)


if __name__ == "__main__":
    unittest.main()
