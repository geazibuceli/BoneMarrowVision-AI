import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location(
    "dependency_audit", Path(__file__).resolve().parents[1] / "audit_locked_requirements.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class DependencyAuditTests(unittest.TestCase):
    def run_response(self, results):
        response = io.BytesIO(json.dumps({"results": results}).encode())
        with patch.object(audit, "pins", return_value={"torch": "2.14.0+cpu"}), \
             patch.object(audit, "urlopen", return_value=response) as query, \
             redirect_stdout(io.StringIO()):
            audit.audit()
        return json.loads(query.call_args.args[0].data)

    def test_queries_local_wheel_version_without_build_suffix(self):
        request = self.run_response([{}])
        self.assertEqual(request["queries"], [
            {"package": {"name": "torch", "ecosystem": "PyPI"}, "version": "2.14.0"}
        ])

    def test_advisories_fail_the_check(self):
        with self.assertRaisesRegex(SystemExit, "GHSA-example"):
            self.run_response([{"vulns": [{"id": "GHSA-example"}]}])

    def test_missing_results_fail_the_check(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            self.run_response([])

    def test_pagination_is_not_silently_ignored(self):
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            self.run_response([{"next_page_token": "more"}])

    def test_network_failure_is_not_reported_as_clean(self):
        with patch.object(audit, "pins", return_value={"example": "1.0"}), \
             patch.object(audit, "urlopen", side_effect=TimeoutError), \
             self.assertRaises(TimeoutError):
            audit.audit()
