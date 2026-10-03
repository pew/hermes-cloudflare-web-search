"""Contract tests for the Cloudflare Web Search Hermes backend.

Run against a Hermes source checkout (see AGENTS.md)::

    PYTHONPATH="$HERMES_SOURCE:$PWD" uv run --project "$HERMES_SOURCE" python -m unittest discover -s tests -v

Uses a standard-library HTTP double; no network access or real credentials.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import provider as cf
from agent.web_search_provider import WebSearchProvider

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = "0123456789abcdef0123456789abcdef"
TOKEN = "test-token"


_real_sleep = time.sleep


class _Handler(BaseHTTPRequestHandler):
    server_version = "CFDouble/1.0"

    def log_message(self, *args):  # silence
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        self.server.requests.append({
            "path": self.path,
            "headers": {k.lower(): v for k, v in self.headers.items()},
            "body": json.loads(raw or b"{}"),
        })
        status, payload, delay = self.server.responses.pop(0) if self.server.responses else (200, {"items": []}, 0)
        if delay:
            _real_sleep(delay)
        data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


class _Double:
    def __init__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.requests = []
        self.server.responses = []
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    @property
    def base(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/client/v4"

    def queue(self, status, payload, delay=0.0):
        self.server.responses.append((status, payload, delay))

    @property
    def requests(self):
        return self.server.requests

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _sample_items():
    return [
        {
            "url": "https://example.com/a",
            "title": "Page A",
            "description": "Page A\n\n   Some   useful\ttext about A.\n\n\n\nMore.",
        },
        {"url": "https://example.com/b", "title": "", "description": "B text"},
        {"url": "https://example.com/a", "title": "Dup", "description": "dup"},
        {"title": "No URL", "description": "skip me"},
        {
            "url": "https://example.com/c",
            "title": "Page C",
            "description": "C text",
            "imageUrl": "https://example.com/c.png",
            "lastModifiedDate": "2026-09-30",
        },
    ]


class ProviderTestCase(unittest.TestCase):
    def setUp(self):
        self.double = _Double()
        self.addCleanup(self.double.close)
        self.env = {cf.ACCOUNT_ENV: ACCOUNT, cf.TOKEN_ENV: TOKEN}
        env_patch = patch.object(cf, "env_value", side_effect=lambda name: self.env.get(name, "").strip())
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.config_mock = {}
        self.config_patch = patch.object(cf, "load_web_config", side_effect=lambda: dict(self.config_mock))
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        sleep_patch = patch.object(cf.time, "sleep", lambda s: None)
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)
        self.provider = cf.CloudflareWebSearchProvider(api_base=self.double.base)


class ContractTests(ProviderTestCase):
    def test_is_web_search_provider_with_expected_identity(self):
        self.assertIsInstance(self.provider, WebSearchProvider)
        self.assertEqual(self.provider.name, "cloudflare")
        self.assertEqual(self.provider.display_name, "Cloudflare Web Search")
        self.assertTrue(self.provider.supports_search())
        self.assertFalse(self.provider.supports_extract())
        self.assertFalse(self.provider.is_keyless_available())

    def test_is_available_requires_account_and_token(self):
        self.assertTrue(self.provider.is_available())
        del self.env[cf.TOKEN_ENV]
        self.assertFalse(self.provider.is_available())
        self.env[cf.TOKEN_OVERRIDE_ENV] = "scoped"
        self.assertTrue(self.provider.is_available())
        del self.env[cf.TOKEN_OVERRIDE_ENV]
        self.env[cf.TOKEN_SEARCH_OVERRIDE_ENV] = "scoped-search"
        self.assertTrue(self.provider.is_available())
        del self.env[cf.ACCOUNT_ENV]
        self.assertFalse(self.provider.is_available())
        self.env[cf.ACCOUNT_OVERRIDE_ENV] = "scoped-account"
        self.assertTrue(self.provider.is_available())
        del self.env[cf.ACCOUNT_OVERRIDE_ENV]
        self.env[cf.ACCOUNT_SEARCH_OVERRIDE_ENV] = "scoped-account-search"
        self.assertTrue(self.provider.is_available())

    def test_is_available_never_raises_and_makes_no_requests(self):
        with patch.object(cf, "env_value", side_effect=RuntimeError("boom")):
            self.assertFalse(self.provider.is_available())
        self.assertEqual(self.double.requests, [])

    def test_setup_schema_lists_required_env_vars(self):
        schema = self.provider.get_setup_schema()
        self.assertEqual(schema["name"], "Cloudflare Web Search")
        self.assertEqual([e["key"] for e in schema["env_vars"]], [cf.ACCOUNT_ENV, cf.TOKEN_ENV])

    def test_manifest_matches_provider(self):
        from hermes_cli.plugins_manifest import parse_manifest_file

        manifest = parse_manifest_file(ROOT / "plugin.yaml", ROOT, "user", "")
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.kind, "backend")
        self.assertEqual(manifest.version, cf.__version__)
        self.assertEqual([e["name"] for e in manifest.requires_env], [cf.ACCOUNT_ENV, cf.TOKEN_ENV])

        # Also check raw manifest text contains provider name declaration
        raw_text = (ROOT / "plugin.yaml").read_text()
        self.assertIn(f"- {self.provider.name}", raw_text)


class RequestTests(ProviderTestCase):
    def test_default_request_shape(self):
        self.double.queue(200, {"items": [], "metadata": {"requestId": "r1"}})
        result = self.provider.search("  hello world  ", 3)
        self.assertEqual(result, {"success": True, "data": {"web": []}})
        req = self.double.requests[0]
        self.assertEqual(req["path"], f"/client/v4/accounts/{ACCOUNT}/ai/websearch/")
        self.assertEqual(req["headers"]["authorization"], f"Bearer {TOKEN}")
        self.assertEqual(req["headers"]["content-type"], "application/json")
        self.assertIn("hermes-cloudflare-web-search/", req["headers"]["user-agent"])
        self.assertEqual(
            req["body"], {"query": "hello world", "limit": 3, "options": {"gateway": {"id": "default"}}}
        )

    def test_optional_settings_are_sent(self):
        self.env.update({
            cf.SEARCH_PROVIDER_ENV: "EXA",
            cf.GATEWAY_ENV: "my-gw",
            cf.BYOK_ENV: "team_key-1",
            cf.TOKEN_OVERRIDE_ENV: "scoped-token",
        })
        self.double.queue(200, {"items": []})
        self.provider.search("q", 5)
        req = self.double.requests[0]
        self.assertEqual(req["headers"]["authorization"], "Bearer scoped-token")
        self.assertEqual(req["body"], {
            "query": "q", "limit": 5, "provider": "exa", "byokAlias": "team_key-1",
            "options": {"gateway": {"id": "my-gw"}},
        })

    def test_limit_is_clamped_to_api_range(self):
        for requested, sent in ((50, 10), (0, 1), (-3, 1), ("7", 7), ("bad", 5), (None, 5)):
            self.double.queue(200, {"items": []})
            self.provider.search("q", requested)
            self.assertEqual(self.double.requests[-1]["body"]["limit"], sent, requested)

    def test_long_query_is_truncated_to_api_maximum(self):
        self.double.queue(200, {"items": []})
        self.provider.search("x" * 3000, 5)
        self.assertEqual(len(self.double.requests[0]["body"]["query"]), cf.MAX_QUERY_CHARS)

    def test_account_id_is_path_quoted(self):
        self.env[cf.ACCOUNT_ENV] = "../../evil?x=1"
        self.double.queue(200, {"items": []})
        self.provider.search("q", 1)
        self.assertEqual(self.double.requests[0]["path"], "/client/v4/accounts/..%2F..%2Fevil%3Fx%3D1/ai/websearch/")

    def test_account_and_token_overrides_take_precedence(self):
        self.env[cf.ACCOUNT_ENV] = "base-account"
        self.env[cf.TOKEN_ENV] = "base-token"
        self.env[cf.ACCOUNT_OVERRIDE_ENV] = "override-account"
        self.env[cf.TOKEN_OVERRIDE_ENV] = "override-token"
        self.double.queue(200, {"items": []})
        self.provider.search("q", 1)
        self.assertEqual(self.double.requests[0]["path"], "/client/v4/accounts/override-account/ai/websearch/")
        self.assertEqual(self.double.requests[0]["headers"]["authorization"], "Bearer override-token")


class ResponseTests(ProviderTestCase):
    def test_items_are_normalized(self):
        self.double.queue(200, {"items": _sample_items(), "metadata": {"query": "q", "requestId": "r", "latencyMs": 5}})
        result = self.provider.search("q", 10)
        self.assertTrue(result["success"])
        rows = result["data"]["web"]
        self.assertEqual([r["url"] for r in rows], ["https://example.com/a", "https://example.com/b", "https://example.com/c"])
        self.assertEqual([r["position"] for r in rows], [1, 2, 3])
        for row in rows:
            self.assertEqual(list(row.keys()), ["title", "url", "description", "position"])
        self.assertEqual(rows[0]["title"], "Page A")
        self.assertEqual(rows[0]["description"], "Some useful text about A.\n\nMore.")
        self.assertEqual(rows[1]["title"], "https://example.com/b")

    def test_items_fallback_to_snippet_or_text(self):
        items = [
            {"url": "https://example.com/1", "title": "Doc 1", "snippet": "Snippet content"},
            {"url": "https://example.com/2", "title": "Doc 2", "text": "Text content"},
        ]
        self.double.queue(200, {"items": items})
        res = self.provider.search("q", 5)
        self.assertTrue(res["success"])
        rows = res["data"]["web"]
        self.assertEqual(rows[0]["description"], "Snippet content")
        self.assertEqual(rows[1]["description"], "Text content")

    def test_result_count_respects_limit(self):
        items = [{"url": f"https://e.com/{i}", "title": str(i), "description": ""} for i in range(10)]
        self.double.queue(200, {"items": items})
        self.assertEqual(len(self.provider.search("q", 4)["data"]["web"]), 4)

    def test_descriptions_are_capped(self):
        long_text = ("word " * 2000).strip()
        self.double.queue(200, {"items": [{"url": "https://e.com", "title": "t", "description": long_text}]})
        desc = self.provider.search("q", 1)["data"]["web"][0]["description"]
        self.assertLessEqual(len(desc), cf.DEFAULT_MAX_DESCRIPTION_CHARS + 1)
        self.assertTrue(desc.endswith("\u2026"))
        self.assertFalse(desc[:-1].endswith(" "))

        self.env[cf.MAX_DESCRIPTION_ENV] = "0"
        self.double.queue(200, {"items": [{"url": "https://e.com", "title": "t", "description": long_text}]})
        self.assertEqual(self.provider.search("q", 1)["data"]["web"][0]["description"], long_text)

    def test_v4_result_envelope_is_accepted(self):
        self.double.queue(200, {"success": True, "result": {"items": [{"url": "https://e.com", "title": "E"}]}})
        rows = self.provider.search("q", 1)["data"]["web"]
        self.assertEqual(rows, [{"title": "E", "url": "https://e.com", "description": "", "position": 1}])

    def test_missing_items_is_an_error(self):
        self.double.queue(200, {"metadata": {}})
        result = self.provider.search("q", 1)
        self.assertFalse(result["success"])
        self.assertIn("missing 'items'", result["error"])

    def test_non_json_success_is_an_error(self):
        self.double.queue(200, "<html>oops</html>")
        result = self.provider.search("q", 1)
        self.assertEqual(result, {"success": False, "error": "Could not parse Cloudflare Web Search response as JSON"})

    def test_search_query_is_not_logged(self):
        self.double.queue(200, {"items": []})
        with self.assertLogs(cf.logger, level="INFO") as logs:
            self.assertTrue(self.provider.search("private-search-marker")["success"])
        self.assertNotIn("private-search-marker", "\n".join(logs.output))


class ErrorTests(ProviderTestCase):
    def assertFailure(self, result, *needles):
        self.assertEqual(set(result.keys()), {"success", "error"})
        self.assertFalse(result["success"])
        for needle in needles:
            self.assertIn(needle, result["error"])

    def test_missing_credentials_fail_without_request(self):
        self.env.clear()
        self.assertFailure(self.provider.search("q"), "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are not set")
        self.env[cf.ACCOUNT_ENV] = ACCOUNT
        self.assertFailure(self.provider.search("q"), "CLOUDFLARE_API_TOKEN is not set")
        self.assertEqual(self.double.requests, [])

    def test_invalid_settings_fail_without_request(self):
        cases = [
            ({cf.SEARCH_PROVIDER_ENV: "bing"}, "ceramic, exa, linkup"),
            ({cf.BYOK_ENV: "bad alias!"}, cf.BYOK_ENV),
            ({cf.MAX_DESCRIPTION_ENV: "lots"}, cf.MAX_DESCRIPTION_ENV),
            ({cf.MAX_DESCRIPTION_ENV: "-1"}, ">= 0"),
            ({cf.TIMEOUT_ENV: "0"}, cf.TIMEOUT_ENV),
        ]
        for extra, needle in cases:
            env = {cf.ACCOUNT_ENV: ACCOUNT, cf.TOKEN_ENV: TOKEN, **extra}
            with patch.object(cf, "env_value", side_effect=lambda n, env=env: env.get(n, "")):
                self.assertFailure(self.provider.search("q"), needle)
        self.assertEqual(self.double.requests, [])

    def test_non_finite_timeouts_fail_without_request(self):
        for timeout in ("nan", "inf", "-inf", "1e999"):
            with self.subTest(timeout=timeout):
                self.env[cf.TIMEOUT_ENV] = timeout
                self.assertFailure(self.provider.search("q"), cf.TIMEOUT_ENV, "finite")
        self.assertEqual(self.double.requests, [])

    def test_empty_query_fails_without_request(self):
        self.assertFailure(self.provider.search("   "), "must not be empty")
        self.assertFailure(self.provider.search(None), "must not be empty")
        self.assertEqual(self.double.requests, [])

    def test_v4_validation_error(self):
        self.double.queue(400, {
            "success": False, "result": None,
            "errors": [{"code": 7000, "message": "Invalid web search request body"}],
            "messages": [{"code": "too_big", "path": ["limit"], "message": "Number must be less than or equal to 10"}],
        })
        self.assertFailure(
            self.provider.search("q"), "HTTP 400", "Invalid web search request body (code 7000)",
            "limit: Number must be less than or equal to 10",
        )

    def test_auth_error_has_permission_hint(self):
        self.double.queue(400, {"success": False, "errors": [{"code": 9106, "message": "Authentication failed (status: 400)"}]})
        self.assertFailure(self.provider.search("q"), "Authentication failed", "Workers AI > Read", "AI Gateway > Read")

    def test_gateway_error_shape_and_hint(self):
        self.env[cf.GATEWAY_ENV] = "missing-gw"
        self.double.queue(400, {
            "success": False, "result": [], "messages": [],
            "error": [{"code": 2001, "message": "Please configure AI Gateway in the Cloudflare dashboard"}],
            "name": "AiGatewayError", "internalCode": 2001,
            "message": "Please configure AI Gateway in the Cloudflare dashboard",
        })
        result = self.provider.search("q")
        self.assertFailure(result, "Please configure AI Gateway", "'missing-gw'", cf.GATEWAY_ENV)
        self.assertEqual(result["error"].count("Please configure AI Gateway"), 1)

    def test_ok_false_error_shape_and_byok_hint(self):
        self.env[cf.BYOK_ENV] = "nope"
        self.double.queue(400, {"ok": False, "error": {
            "category": "credential", "code": "web_search_byok_not_configured", "status": 400,
            "retryable": False, "gatewayRequestId": "gw-1",
        }})
        self.assertFailure(self.provider.search("q"), "web_search_byok_not_configured", "gw-1", "alias 'nope'")

    def test_rate_limit_is_not_retried(self):
        self.double.queue(429, {"success": False, "errors": [{"code": 971, "message": "Too many requests"}]})
        self.assertFailure(self.provider.search("q"), "HTTP 429", "rate limited")
        self.assertEqual(len(self.double.requests), 1)

    def test_server_error_is_retried_once(self):
        self.double.queue(503, "upstream down")
        self.double.queue(200, {"items": [{"url": "https://e.com", "title": "E"}]})
        result = self.provider.search("q")
        self.assertTrue(result["success"])
        self.assertEqual(len(self.double.requests), 2)

    def test_persistent_server_error(self):
        self.double.queue(502, "bad gateway")
        self.double.queue(502, "bad gateway")
        self.assertFailure(self.provider.search("q"), "HTTP 502", "bad gateway")
        self.assertEqual(len(self.double.requests), 2)

    def test_timeout(self):
        self.env[cf.TIMEOUT_ENV] = "0.2"
        self.double.queue(200, {"items": []}, delay=1.0)
        self.double.queue(200, {"items": []}, delay=1.0)
        self.assertFailure(self.provider.search("q"), "timed out after 0.2s")

    def test_unreachable(self):
        self.env[cf.TIMEOUT_ENV] = "0.2"
        provider = cf.CloudflareWebSearchProvider(api_base="http://127.0.0.1:1/client/v4")
        self.assertFailure(provider.search("q"), "Could not reach Cloudflare Web Search")

    def test_unexpected_exception_is_wrapped(self):
        with patch.object(cf.httpx, "post", side_effect=RuntimeError("kaboom")):
            self.assertFailure(self.provider.search("q"), "search failed: kaboom")

    def test_interrupt_short_circuits(self):
        with patch.object(cf, "_interrupted", return_value=True):
            self.assertEqual(self.provider.search("q"), {"success": False, "error": "Interrupted"})
        self.assertEqual(self.double.requests, [])

    def test_interrupt_during_retry_delay_stops_next_request(self):
        interrupted = False

        def interrupt(_):
            nonlocal interrupted
            interrupted = True

        self.double.queue(503, "upstream down")
        self.double.queue(200, {"items": []})
        with patch.object(cf, "_interrupted", side_effect=lambda: interrupted), \
                patch.object(cf.time, "sleep", side_effect=interrupt):
            self.assertEqual(self.provider.search("q"), {"success": False, "error": "Interrupted"})
        self.assertEqual(len(self.double.requests), 1)

    def test_interrupt_during_request_discards_results(self):
        self.double.queue(200, {"items": []})
        real_post = cf.httpx.post
        interrupted = False

        def post(*args, **kwargs):
            nonlocal interrupted
            response = real_post(*args, **kwargs)
            interrupted = True
            return response

        with patch.object(cf, "_interrupted", side_effect=lambda: interrupted), \
                patch.object(cf.httpx, "post", side_effect=post):
            self.assertEqual(self.provider.search("q"), {"success": False, "error": "Interrupted"})

    def test_malformed_token_is_not_exposed_by_httpx(self):
        self.env[cf.TOKEN_ENV] = "synthetic-secret-marker\ninvalid"
        result = self.provider.search("q")
        self.assertFalse(result["success"])
        self.assertNotIn("synthetic-secret-marker", result["error"])

    def test_token_echo_in_upstream_error_is_redacted(self):
        self.double.queue(400, {"errors": [{"message": f"Rejected credential {TOKEN}"}]})
        result = self.provider.search("q")
        self.assertFalse(result["success"])
        self.assertNotIn(TOKEN, result["error"])

    def test_error_body_redacts_token_before_truncation(self):
        self.env[cf.TOKEN_ENV] = "synthetic-sensitive-token-marker-with-extra-length"
        self.double.queue(400, "x" * 280 + self.env[cf.TOKEN_ENV])
        result = self.provider.search("q")
        self.assertFalse(result["success"])
        self.assertNotIn("synthetic-sensitive-", result["error"])

    def test_token_in_unexpected_exception_is_not_returned_or_logged(self):
        with patch.object(cf.httpx, "post", side_effect=RuntimeError(f"Rejected {TOKEN}")), \
                self.assertLogs(cf.logger, level="WARNING") as logs:
            result = self.provider.search("q")
        self.assertFalse(result["success"])
        self.assertNotIn(TOKEN, result["error"])
        self.assertNotIn(TOKEN, "\n".join(logs.output))

    def test_token_is_never_in_error_messages(self):
        self.double.queue(400, {"success": False, "errors": [{"code": 9106, "message": "Authentication failed"}]})
        self.assertNotIn(TOKEN, self.provider.search("q")["error"])


class HelperTests(unittest.TestCase):
    def test_clean_description_strips_title_prefix_only_at_boundary(self):
        self.assertEqual(cf.clean_description("Title\n\nBody", "Title"), "Body")
        self.assertEqual(cf.clean_description("Title - Body", "Title"), "Body")
        self.assertEqual(cf.clean_description("Titles are great", "Title"), "Titles are great")
        self.assertEqual(cf.clean_description("Title", "Title"), "Title")
        self.assertEqual(cf.clean_description(None, "Title"), "")
        self.assertEqual(cf.clean_description("Text", None), "Text")
        self.assertEqual(cf.clean_description("Text", 123), "Text")

    def test_env_value_reads_process_environment(self):
        with patch.dict("os.environ", {"CF_WS_TEST_VAR": "  value  "}):
            self.assertEqual(cf.env_value("CF_WS_TEST_VAR"), "value")


class RegistrationTests(unittest.TestCase):
    def test_register_loads_like_hermes_directory_plugin(self):
        """Import the repo root as ``hermes_plugins.<slug>`` (as Hermes' loader does) and register."""
        name = "hermes_plugins_test.cloudflare_web_search"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self.addCleanup(lambda: [sys.modules.pop(k) for k in list(sys.modules) if k.startswith(name)])
        spec.loader.exec_module(module)

        registered = []

        class Ctx:
            def register_web_search_provider(self, provider):
                registered.append(provider)

        module.register(Ctx())
        self.assertEqual(len(registered), 1)
        self.assertEqual(registered[0].name, "cloudflare")
        self.assertIsInstance(registered[0], WebSearchProvider)


class ConfigYamlTests(ProviderTestCase):
    def test_config_read_failure_is_reported_without_search(self):
        self.config_patch.stop()
        with patch("hermes_cli.config.load_config_readonly", side_effect=ValueError("broken config")), \
                patch("hermes_cli.config.load_config", side_effect=ValueError("broken config")):
            result = self.provider.search("q")
            self.assertFalse(result["success"])
            self.assertIn("configuration", result["error"])
            self.assertFalse(self.provider.is_available())
        self.assertEqual(self.double.requests, [])

    def test_account_id_configured_in_config_yaml_without_env(self):
        del self.env[cf.ACCOUNT_ENV]
        self.config_mock["account_id"] = "cfg-account-123"
        self.assertTrue(self.provider.is_available())
        self.double.queue(200, {"items": []})
        self.provider.search("q", 1)
        req = self.double.requests[0]
        self.assertEqual(req["path"], "/client/v4/accounts/cfg-account-123/ai/websearch/")

    def test_all_custom_settings_configured_in_yaml(self):
        self.config_mock.update({
            "account_id": "cfg-account-xyz",
            "gateway_id": "my-custom-gateway",
            "provider": "linkup",
            "byok_alias": "custom-alias",
            "timeout": 42.5,
            "max_description_chars": 150,
        })
        settings = self.provider._settings()
        self.assertEqual(settings["account_id"], "cfg-account-xyz")
        self.assertEqual(settings["gateway_id"], "my-custom-gateway")
        self.assertEqual(settings["search_provider"], "linkup")
        self.assertEqual(settings["byok_alias"], "custom-alias")
        self.assertEqual(settings["timeout"], 42.5)
        self.assertEqual(settings["max_chars"], 150)

        self.double.queue(200, {"items": []})
        self.provider.search("q", 2)
        req = self.double.requests[0]
        self.assertEqual(req["path"], "/client/v4/accounts/cfg-account-xyz/ai/websearch/")
        self.assertEqual(req["body"], {
            "query": "q",
            "limit": 2,
            "provider": "linkup",
            "byokAlias": "custom-alias",
            "options": {"gateway": {"id": "my-custom-gateway"}},
        })

    def test_env_override_takes_precedence_over_config_yaml(self):
        self.config_mock.update({
            "account_id": "yaml-account",
            "gateway_id": "yaml-gateway",
            "provider": "ceramic",
        })
        self.env.update({
            cf.ACCOUNT_OVERRIDE_ENV: "env-override-account",
            cf.GATEWAY_ENV: "env-override-gateway",
            cf.SEARCH_PROVIDER_ENV: "exa",
        })
        settings = self.provider._settings()
        self.assertEqual(settings["account_id"], "env-override-account")
        self.assertEqual(settings["gateway_id"], "env-override-gateway")
        self.assertEqual(settings["search_provider"], "exa")

    def test_explicit_config_injection(self):
        custom_provider = cf.CloudflareWebSearchProvider(
            api_base=self.double.base,
            config={"account_id": "injected-acc", "gateway_id": "injected-gw", "provider": "exa"},
        )
        del self.env[cf.ACCOUNT_ENV]
        self.assertTrue(custom_provider.is_available())
        settings = custom_provider._settings()
        self.assertEqual(settings["account_id"], "injected-acc")
        self.assertEqual(settings["gateway_id"], "injected-gw")
        self.assertEqual(settings["search_provider"], "exa")

    def test_load_web_config_parses_web_cloudflare_and_plugin_entries(self):
        fake_config = {
            "plugins": {
                "entries": {
                    "cloudflare-web-search": {
                        "settings": {"timeout": 15, "provider": "ceramic"},
                        "account_id": "entries-acc",
                    }
                }
            },
            "web": {
                "cloudflare": {
                    "account_id": "web-cf-acc",
                    "gateway_id": "web-cf-gw",
                }
            },
        }
        self.config_patch.stop()
        try:
            with patch("hermes_cli.config.load_config_readonly", return_value=fake_config):
                loaded = cf.load_web_config()
                self.assertEqual(loaded["timeout"], 15)
                self.assertEqual(loaded["provider"], "ceramic")
                # web.cloudflare overrides plugins.entries
                self.assertEqual(loaded["account_id"], "web-cf-acc")
                self.assertEqual(loaded["gateway_id"], "web-cf-gw")
        finally:
            self.config_patch.start()

    def test_load_web_config_parses_plugins_top_level_entry(self):
        fake_config = {
            "plugins": {
                "cloudflare-web-search": {
                    "account_id": "plugin-acc",
                    "settings": {
                        "provider": "exa",
                        "timeout": 30,
                    },
                }
            }
        }
        self.config_patch.stop()
        try:
            with patch("hermes_cli.config.load_config_readonly", return_value=fake_config):
                loaded = cf.load_web_config()
                self.assertEqual(loaded["account_id"], "plugin-acc")
                self.assertEqual(loaded["provider"], "exa")
                self.assertEqual(loaded["timeout"], 30)
        finally:
            self.config_patch.start()

    def test_search_alias_env_vars_work(self):
        del self.env[cf.TOKEN_ENV]
        del self.env[cf.ACCOUNT_ENV]
        self.env[cf.ACCOUNT_SEARCH_OVERRIDE_ENV] = "alias-acc"
        self.env[cf.TOKEN_SEARCH_OVERRIDE_ENV] = "alias-token"
        self.env[cf.SEARCH_PROVIDER_ALIAS_ENV] = "linkup"
        self.env[cf.GATEWAY_ALIAS_ENV] = "alias-gw"
        self.env[cf.BYOK_ALIAS_ENV] = "alias-byok"
        self.env[cf.TIMEOUT_SEARCH_ENV] = "15"
        self.env[cf.MAX_DESCRIPTION_ALIAS_ENV] = "500"
        self.assertTrue(self.provider.is_available())
        settings = self.provider._settings()
        self.assertEqual(settings["account_id"], "alias-acc")
        self.assertEqual(settings["token"], "alias-token")
        self.assertEqual(settings["search_provider"], "linkup")
        self.assertEqual(settings["gateway_id"], "alias-gw")
        self.assertEqual(settings["byok_alias"], "alias-byok")
        self.assertEqual(settings["timeout"], 15.0)
        self.assertEqual(settings["max_chars"], 500)


if __name__ == "__main__":
    unittest.main()
