"""Opt-in real sandbox smoke test; synthetic data never enters reports/."""
import json
import os
import unittest
from unittest.mock import patch

from agents import AUDIT_PATH, FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, NonInteractiveSandbox, make_audit_tools
from research import FINALIZER_SOURCE, VALIDATOR_SOURCE, validate_audit
from sandbox import download, open_sandbox, upload


@unittest.skipUnless(os.getenv("LAB_RUN_SANDBOX_TEST") == "1", "opt-in real Daytona/Docker sandbox")
class SandboxSmoke(unittest.TestCase):
    def test_execute_finalize_validate_download_and_cleanup(self):
        sources = [{"n": 7, "id": "fixture", "url": "https://example.org/fixture", "title": "Synthetic fixture",
                    "date": "", "source": "web"}]
        with open_sandbox() as backend:
            backend = NonInteractiveSandbox(backend)
            eof = backend.execute('python3 -c "import sys; print(len(sys.stdin.read()))"')
            self.assertEqual(eof.exit_code, 0, eof.output)
            self.assertEqual(eof.output.strip(), "0")
            result = backend.execute(f"mkdir -p {WORKDIR}/research {WORKDIR}/report")
            self.assertEqual(result.exit_code, 0)
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                             REPORT_PATH: b"# Synthetic fixture\nClaim [7].\n",
                             SOURCES_PATH: json.dumps(sources).encode()})
            finalized = backend.execute(f"python3 {FINALIZER_PATH}")
            self.assertEqual(finalized.exit_code, 0, finalized.output)
            validated = backend.execute(f"python3 {VALIDATOR_PATH}")
            self.assertEqual(validated.exit_code, 0, validated.output)
            self.assertIn("OK: 1 sources", validated.output)
            files = download(backend, [REPORT_PATH, SOURCES_PATH])
            self.assertIn(b"Claim [1]", files[REPORT_PATH])
            self.assertIn(b"## References", files[REPORT_PATH])
            self.assertEqual(json.loads(files[SOURCES_PATH])[0]["n"], 1)
            prepare, record = make_audit_tools(backend)
            self.assertEqual(len(json.loads(prepare.invoke({}))["sources"]), 1)
            url = sources[0]["url"]
            pages = {url: "Synthetic source evidence for this isolated sandbox regression test."}
            with patch("agents.get_fetched_pages", return_value=pages):
                self.assertIn("Recorded [1]", record.invoke({"n": 1, "status": "SUPPORTED", "reason": "Synthetic fixture only", "evidence": "Synthetic source evidence"}))
                self.assertTrue(record.invoke({"n": 1, "status": "SUPPORTED", "reason": "Bad excerpt", "evidence": "invented evidence excerpt"}).startswith("ERROR:"))
                self.assertIn("Recorded [1]", record.invoke({"n": 1, "status": "SUPPORTED", "reason": "Automatic literal witness"}))
            audit = download(backend, [AUDIT_PATH])[AUDIT_PATH]
            self.assertEqual(len(validate_audit(audit, json.loads(files[SOURCES_PATH]), files[REPORT_PATH].decode(), pages)), 1)
            print("\nReal sandbox: execute, finalizer, validator and download passed; no report saved.", flush=True)


if __name__ == "__main__":
    unittest.main()
