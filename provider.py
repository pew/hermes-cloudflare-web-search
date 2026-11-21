"""Cloudflare Web Search API backend for Hermes' ``web_search`` tool.

Calls ``POST /accounts/{account_id}/ai/websearch/`` (routed through an AI Gateway)
and normalizes the result into Hermes' fixed search envelope::

    {"success": True, "data": {"web": [{"title", "url", "description", "position"}, ...]}}

Search only — Cloudflare's Web Search API has no extract endpoint, so pair this
backend with an extract provider (``web.extract_backend``) for ``web_extract``.

Environment (``os.environ`` first, then ``~/.hermes/.env``):

* ``CLOUDFLARE_ACCOUNT_ID`` (required). ``CLOUDFLARE_WEB_SEARCH_ACCOUNT_ID`` takes
  precedence when set.
* ``CLOUDFLARE_API_TOKEN`` (required; Account > Workers AI > Read and Account > AI Gateway > Read).
  ``CLOUDFLARE_WEB_SEARCH_API_TOKEN`` takes precedence when set, so a narrowly
  scoped token can be used without touching a shared ``CLOUDFLARE_API_TOKEN``.
* ``CLOUDFLARE_WEB_SEARCH_PROVIDER`` — ``ceramic`` | ``exa`` | ``linkup`` (default: Cloudflare's default, Ceramic.ai)
* ``CLOUDFLARE_WEB_SEARCH_GATEWAY_ID`` — AI Gateway id (default ``default``)
* ``CLOUDFLARE_WEB_SEARCH_BYOK_ALIAS`` — alias of a provider key stored on the gateway
* ``CLOUDFLARE_WEB_SEARCH_MAX_DESCRIPTION_CHARS`` — per-result description cap (default 2000, ``0`` = unlimited)
* ``CLOUDFLARE_WEB_SEARCH_TIMEOUT`` — request timeout in seconds (default 20)
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

PROVIDER_NAME = "cloudflare"
DISPLAY_NAME = "Cloudflare Web Search"
VENDOR = "Cloudflare Web Search"

API_BASE = "https://api.cloudflare.com/client/v4"

ACCOUNT_ENV = "CLOUDFLARE_ACCOUNT_ID"
ACCOUNT_OVERRIDE_ENV = "CLOUDFLARE_WEB_SEARCH_ACCOUNT_ID"
ACCOUNT_SEARCH_OVERRIDE_ENV = "CLOUDFLARE_SEARCH_ACCOUNT_ID"
TOKEN_ENV = "CLOUDFLARE_API_TOKEN"
TOKEN_OVERRIDE_ENV = "CLOUDFLARE_WEB_SEARCH_API_TOKEN"
TOKEN_SEARCH_OVERRIDE_ENV = "CLOUDFLARE_SEARCH_API_TOKEN"
SEARCH_PROVIDER_ENV = "CLOUDFLARE_WEB_SEARCH_PROVIDER"
SEARCH_PROVIDER_ALIAS_ENV = "CLOUDFLARE_SEARCH_PROVIDER"
GATEWAY_ENV = "CLOUDFLARE_WEB_SEARCH_GATEWAY_ID"
GATEWAY_ALIAS_ENV = "CLOUDFLARE_SEARCH_GATEWAY_ID"
BYOK_ENV = "CLOUDFLARE_WEB_SEARCH_BYOK_ALIAS"
BYOK_ALIAS_ENV = "CLOUDFLARE_SEARCH_BYOK_ALIAS"
MAX_DESCRIPTION_ENV = "CLOUDFLARE_WEB_SEARCH_MAX_DESCRIPTION_CHARS"
MAX_DESCRIPTION_ALIAS_ENV = "CLOUDFLARE_SEARCH_MAX_SNIPPET_LENGTH"
TIMEOUT_ENV = "CLOUDFLARE_WEB_SEARCH_TIMEOUT"
TIMEOUT_SEARCH_ENV = "CLOUDFLARE_SEARCH_TIMEOUT"

SEARCH_PROVIDERS = ("ceramic", "exa", "linkup")
DEFAULT_GATEWAY_ID = "default"
DEFAULT_MAX_DESCRIPTION_CHARS = 2000
DEFAULT_TIMEOUT = 20.0
DEFAULT_LIMIT = 5
MAX_LIMIT = 10  # API maximum ("limit" > 10 is rejected with HTTP 400)
MAX_QUERY_CHARS = 1024  # API maximum query length

TOKEN_HELP = "the token needs Account > Workers AI > Read and Account > AI Gateway > Read permissions"

_BYOK_ALIAS_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_RETRYABLE_STATUSES = frozenset({500, 502, 503, 504})
_RETRY_DELAY_SECONDS = 0.5
_MAX_ERROR_DETAIL_CHARS = 500
_MAX_RAW_ERROR_CHARS = 300


class ConfigError(ValueError):
    """Invalid or missing plugin configuration (message is user-facing)."""


# --------------------------------------------------------------------------- helpers
def env_value(name: str) -> str:
    """Config-aware env lookup: Hermes' ``get_provider_env`` (``os.environ`` then
    ``~/.hermes/.env``, profile-scope aware), falling back to ``os.getenv``."""
    try:
        from agent.web_search_provider import get_provider_env
    except ImportError:  # very old Hermes without the helper
        return (os.getenv(name) or "").strip()
    return get_provider_env(name)


def _interrupted() -> bool:
    try:
        from tools.interrupt import is_interrupted
    except Exception:  # noqa: BLE001 — optional in stripped/test environments
        return False
    try:
        return bool(is_interrupted())
    except Exception:  # noqa: BLE001
        return False


def search_fail(error: str) -> Dict[str, Any]:
    return {"success": False, "error": error}


def redact_error(message: str, token: str) -> str:
    """Strip the credential, including escaped/URL-encoded forms used by HTTP errors."""
    if token:
        candidates = (
            token,
            repr(token)[1:-1],
            ascii(token)[1:-1],
            repr(token.encode())[2:-1],
            quote(token, safe=""),
        )
        for value in dict.fromkeys(candidates):
            if value:
                message = message.replace(value, "[redacted]")
    return message


def clamp_limit(limit: Any) -> int:
    if isinstance(limit, bool):
        return DEFAULT_LIMIT
    try:
        value = int(limit)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_LIMIT
    return max(1, min(value, MAX_LIMIT))


def clean_description(text: Any, title: Any = "", max_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS) -> str:
    """Normalize whitespace, drop a leading repeat of the title (Ceramic.ai prefixes
    page text with it), and cap the length at a word boundary."""
    if not isinstance(text, str):
        return ""
    lines = [re.sub(r"[ \t\f\v\r]+", " ", line).strip() for line in text.split("\n")]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    title_str = title.strip() if isinstance(title, str) else ""
    if title_str and cleaned.startswith(title_str) and len(cleaned) > len(title_str):
        rest = cleaned[len(title_str):]
        if rest[:1] in ("\n", " ", "|", "-", ":", "\u2013", "\u2014"):
            cleaned = rest.lstrip(" \n|-:\u2013\u2014").strip() or cleaned
    if max_chars and max_chars > 0 and len(cleaned) > max_chars:
        cut = cleaned[:max_chars]
        boundary = max(cut.rfind(" "), cut.rfind("\n"))
        if boundary >= int(max_chars * 0.8):
            cut = cut[:boundary]
        cleaned = cut.rstrip() + "\u2026"
    return cleaned


def normalize_items(items: Any, limit: int, max_chars: int) -> List[Dict[str, Any]]:
    """Map Cloudflare ``items[]`` to Hermes rows: skip URL-less entries, de-duplicate
    URLs, fall back to the URL as title, 1-based positions."""
    rows: List[Dict[str, Any]] = []
    seen = set()
    if not isinstance(items, list):
        return rows
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        url = url.strip()
        if url in seen:
            continue
        seen.add(url)
        title = item.get("title")
        title = title.strip() if isinstance(title, str) and title.strip() else url
        desc = ""
        for field in ("description", "snippet", "text"):
            val = item.get(field)
            if isinstance(val, str) and val.strip():
                desc = val
                break
        rows.append({
            "title": title,
            "url": url,
            "description": clean_description(desc, title, max_chars),
            "position": len(rows) + 1,
        })
        if len(rows) >= limit:
            break
    return rows


def _messages(entries: Any) -> List[str]:
    out: List[str] = []
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if isinstance(entry, str) and entry.strip():
            out.append(entry.strip())
        elif isinstance(entry, dict):
            message = entry.get("message")
            if not isinstance(message, str) or not message.strip():
                continue
            path = entry.get("path")
            if isinstance(path, list) and path:
                message = f"{'.'.join(str(p) for p in path)}: {message.strip()}"
            code = entry.get("code")
            if isinstance(code, int) and not isinstance(code, bool):
                out.append(f"{message.strip()} (code {code})")
            else:
                out.append(message.strip())
    return out


def _code_str(value: Any) -> Optional[str]:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    text = str(value).strip()
    return text or None


def error_detail(payload: Any) -> Tuple[str, Optional[str]]:
    """Human-readable detail + machine code from any of the observed error shapes:

    * API v4 envelope: ``{"success": false, "errors": [{"code", "message"}], "messages": [...]}``
    * AI Gateway: ``{"ok": false, "error": {"code": "invalid_web_search_input", "category", ...}}``
    * AiGatewayError: ``{"error": [{"code": 2001, "message"}], "message": ..., "internalCode": 2001}``
    """
    if not isinstance(payload, dict):
        return "", None
    parts: List[str] = []
    code: Optional[str] = None
    errors = payload.get("errors")
    if errors:
        parts += _messages(errors)
        if isinstance(errors, dict):
            code = _code_str(errors.get("code"))
        elif isinstance(errors, list) and errors and isinstance(errors[0], dict):
            code = _code_str(errors[0].get("code"))
    err = payload.get("error")
    if isinstance(err, dict):
        err_code = _code_str(err.get("code"))
        if err_code is not None:
            code = err_code
        message = err.get("message")
        if isinstance(message, str) and message.strip():
            parts.append(message.strip())
        elif code:
            category = err.get("category")
            parts.append(f"{code} ({category})" if category else code)
        if err.get("gatewayRequestId"):
            parts.append(f"gateway request id {err['gatewayRequestId']}")
    elif isinstance(err, list):
        parts += _messages(err)
        if err and isinstance(err[0], dict):
            err_code = _code_str(err[0].get("code"))
            if err_code is not None:
                code = err_code
    elif isinstance(err, str) and err.strip():
        parts.append(err.strip())
    if not parts and isinstance(payload.get("message"), str):
        parts.append(payload["message"].strip())
    for detail in _messages(payload.get("messages")):
        if detail not in parts:
            parts.append(detail)
    if code is None:
        code = _code_str(payload.get("internalCode"))
    return "; ".join(dict.fromkeys(part for part in parts if part)), code


def load_web_config() -> Dict[str, Any]:
    """Read configuration from config.yaml under `web.cloudflare` or `plugins.entries.cloudflare-web-search`."""
    cfg: Dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config_readonly
    except ImportError:  # compatibility with Hermes versions without the read-only helper
        try:
            from hermes_cli.config import load_config as load_config_readonly
        except ImportError:  # config module optional in stripped contexts
            return cfg
    try:
        root = load_config_readonly()
    except Exception as exc:  # noqa: BLE001 — do not silently search with fallback settings
        raise ConfigError(f"Could not read Hermes configuration ({type(exc).__name__}); check config.yaml") from exc

    if not isinstance(root, dict):
        return cfg

    # 1. Check plugins.entries.cloudflare-web-search or plugins.cloudflare-web-search
    plugins = root.get("plugins")
    if isinstance(plugins, dict):
        entries = plugins.get("entries")
        entry = None
        if isinstance(entries, dict):
            entry = entries.get("cloudflare-web-search") or entries.get("cloudflare_web_search")
        if not isinstance(entry, dict):
            entry = plugins.get("cloudflare-web-search") or plugins.get("cloudflare_web_search")
        if isinstance(entry, dict):
            cfg.update({k: v for k, v in entry.items() if k != "settings"})
            if isinstance(entry.get("settings"), dict):
                cfg.update(entry["settings"])

    # 2. Check web.cloudflare (primary section for web search backends)
    web = root.get("web")
    if isinstance(web, dict):
        cf_web = web.get("cloudflare") or web.get("cloudflare_web_search")
        if isinstance(cf_web, dict):
            cfg.update(cf_web)

    return cfg


# --------------------------------------------------------------------------- provider
class CloudflareWebSearchProvider(WebSearchProvider):
    """Search-only Hermes web backend backed by Cloudflare's Web Search API."""

    def __init__(self, api_base: str = API_BASE, config: Optional[Dict[str, Any]] = None) -> None:
        self._api_base = api_base.rstrip("/")
        self._explicit_config = config

    def _get_config(self) -> Dict[str, Any]:
        if self._explicit_config is not None:
            return dict(self._explicit_config)
        return load_web_config()

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return DISPLAY_NAME

    # -- capability flags --------------------------------------------------
    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def is_available(self) -> bool:
        """Cheap, network-free: an account id and a token are configured (via env or config.yaml)."""
        try:
            cfg = self._get_config()
            return bool(self._account_id(cfg) and self._token(cfg))
        except Exception:  # noqa: BLE001 — must never raise (runs on every picker paint)
            return False

    def get_setup_schema(self) -> Dict[str, Any]:
        try:
            cfg = self._get_config()
        except Exception:  # noqa: BLE001 — never hide the provider from `hermes tools`
            cfg = {}

        def _safe_env(name: str) -> str:
            try:
                return env_value(name)
            except Exception:  # noqa: BLE001
                return ""

        env_vars = []
        for names, keys, prompt, url in (
            (
                (ACCOUNT_OVERRIDE_ENV, ACCOUNT_SEARCH_OVERRIDE_ENV, ACCOUNT_ENV),
                ("account_id",),
                "Cloudflare account ID (or set in config.yaml under web.cloudflare.account_id)",
                "https://developers.cloudflare.com/fundamentals/account/find-account-and-zone-ids/",
            ),
            (
                (TOKEN_OVERRIDE_ENV, TOKEN_SEARCH_OVERRIDE_ENV, TOKEN_ENV),
                ("api_token", "token"),
                "Cloudflare API token (Workers AI Read + AI Gateway Read)",
                "https://dash.cloudflare.com/profile/api-tokens",
            ),
        ):
            key = next((name for name in names[:2] if _safe_env(name)), None)
            if key is None:
                yaml_invalid = False
                try:
                    yaml_value = self._config_str(cfg, keys, setting_name=names[-1])
                except ConfigError:
                    yaml_value = ""
                    yaml_invalid = True
                if yaml_value:
                    continue
                key = names[0] if yaml_invalid or not _safe_env(names[-1]) else names[-1]
            env_vars.append({"key": key, "prompt": prompt, "url": url})
        return {
            "name": DISPLAY_NAME,
            "badge": "paid",
            "tag": "Ceramic.ai / Exa / Linkup through AI Gateway \u2014 search only.",
            "env_vars": env_vars,
        }

    # -- configuration -----------------------------------------------------
    @staticmethod
    def _config_str(
        cfg: Dict[str, Any],
        keys: Tuple[str, ...],
        *,
        setting_name: str,
    ) -> str:
        for key in keys:
            val = cfg.get(key)
            if val is None:
                continue
            if isinstance(val, str):
                stripped = val.strip()
                if stripped:
                    return stripped
                continue
            raise ConfigError(f"{setting_name} must be a string")
        return ""

    @staticmethod
    def _config_first(cfg: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
        for key in keys:
            val = cfg.get(key)
            if val is None:
                continue
            if isinstance(val, str) and not val.strip():
                continue
            return val
        return None

    def _account_id(self, cfg: Optional[Dict[str, Any]] = None) -> str:
        if cfg is None:
            cfg = self._get_config()
        # 1. Specific env override: CLOUDFLARE_WEB_SEARCH_ACCOUNT_ID or CLOUDFLARE_SEARCH_ACCOUNT_ID
        override = env_value(ACCOUNT_OVERRIDE_ENV) or env_value(ACCOUNT_SEARCH_OVERRIDE_ENV)
        if override:
            return override
        # 2. config.yaml setting: account_id
        val = self._config_str(cfg, ("account_id",), setting_name=ACCOUNT_ENV)
        if val:
            return val
        # 3. Base env: CLOUDFLARE_ACCOUNT_ID
        return env_value(ACCOUNT_ENV)

    def _token(self, cfg: Optional[Dict[str, Any]] = None) -> str:
        if cfg is None:
            cfg = self._get_config()
        # 1. Specific env override: CLOUDFLARE_WEB_SEARCH_API_TOKEN or CLOUDFLARE_SEARCH_API_TOKEN
        override = env_value(TOKEN_OVERRIDE_ENV) or env_value(TOKEN_SEARCH_OVERRIDE_ENV)
        if override:
            return override
        # 2. config.yaml setting: api_token or token
        val = self._config_str(cfg, ("api_token", "token"), setting_name=TOKEN_ENV)
        if val:
            return val
        # 3. Base env: CLOUDFLARE_API_TOKEN
        return env_value(TOKEN_ENV)

    def _settings(self) -> Dict[str, Any]:
        cfg = self._get_config()
        account_id = self._account_id(cfg)
        token = self._token(cfg)
        missing = []
        if not account_id:
            missing.append(ACCOUNT_ENV)
        if not token:
            missing.append(TOKEN_ENV)
        if missing:
            raise ConfigError(f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not set")

        # Provider: env > config > default "ceramic"
        raw_provider = (
            env_value(SEARCH_PROVIDER_ENV)
            or env_value(SEARCH_PROVIDER_ALIAS_ENV)
            or self._config_str(cfg, ("search_provider", "provider"), setting_name=SEARCH_PROVIDER_ENV)
        )
        search_provider = raw_provider.strip().lower()
        if search_provider and search_provider not in SEARCH_PROVIDERS:
            raise ConfigError(
                f"{SEARCH_PROVIDER_ENV} is invalid; use one of: {', '.join(SEARCH_PROVIDERS)}"
            )

        # Gateway: env > config > default "default"
        raw_gateway = (
            env_value(GATEWAY_ENV)
            or env_value(GATEWAY_ALIAS_ENV)
            or self._config_str(cfg, ("gateway_id", "gateway"), setting_name=GATEWAY_ENV)
            or DEFAULT_GATEWAY_ID
        )
        gateway_id = raw_gateway.strip() or DEFAULT_GATEWAY_ID

        # BYOK alias: env > config > ""
        raw_byok = (
            env_value(BYOK_ENV)
            or env_value(BYOK_ALIAS_ENV)
            or self._config_str(cfg, ("byok_alias", "byok"), setting_name=BYOK_ENV)
        )
        byok_alias = raw_byok.strip()
        if byok_alias and not _BYOK_ALIAS_RE.match(byok_alias):
            raise ConfigError(f"{BYOK_ENV} must match ^[A-Za-z0-9_-]{{1,64}}$")

        # Max description chars: env > config > default 2000
        raw_max_chars: Any = env_value(MAX_DESCRIPTION_ENV) or env_value(MAX_DESCRIPTION_ALIAS_ENV)
        if not raw_max_chars:
            raw_max_chars = self._config_first(cfg, ("max_description_chars", "max_snippet_length", "max_chars"))
        max_chars = self._coerce_int(MAX_DESCRIPTION_ENV, raw_max_chars, DEFAULT_MAX_DESCRIPTION_CHARS, minimum=0)

        # Timeout: env > config > default 20.0
        raw_timeout: Any = env_value(TIMEOUT_ENV) or env_value(TIMEOUT_SEARCH_ENV)
        if not raw_timeout:
            raw_timeout = self._config_first(cfg, ("timeout",))
        timeout = self._coerce_float(TIMEOUT_ENV, raw_timeout, DEFAULT_TIMEOUT)

        return {
            "account_id": account_id,
            "token": token,
            "search_provider": search_provider,
            "gateway_id": gateway_id,
            "byok_alias": byok_alias,
            "max_chars": max_chars,
            "timeout": timeout,
        }

    @staticmethod
    def _coerce_int(name: str, raw: Any, default: int, *, minimum: int) -> int:
        if isinstance(raw, str):
            raw = raw.strip()
        if raw is None or raw == "":
            return default
        if isinstance(raw, (bool, float)):
            raise ConfigError(f"{name} must be an integer")
        try:
            value = int(raw)
        except (ValueError, TypeError, OverflowError):
            raise ConfigError(f"{name} must be an integer")
        if value < minimum:
            raise ConfigError(f"{name} must be >= {minimum}")
        return value

    @staticmethod
    def _coerce_float(name: str, raw: Any, default: float) -> float:
        if isinstance(raw, str):
            raw = raw.strip()
        if raw is None or raw == "":
            return default
        if isinstance(raw, bool):
            raise ConfigError(f"{name} must be a number of seconds")
        try:
            value = float(raw)
        except (ValueError, TypeError, OverflowError):
            raise ConfigError(f"{name} must be a number of seconds")
        if not math.isfinite(value) or value <= 0:
            raise ConfigError(f"{name} must be finite and > 0")
        return value

    def _endpoint(self, account_id: str) -> str:
        return f"{self._api_base}/accounts/{quote(account_id, safe='')}/ai/websearch/"

    @staticmethod
    def build_body(query: str, limit: int, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Build Cloudflare AI Web Search API request payload.

        Cloudflare Web Search endpoint accepts 'query', 'provider', 'limit', 'byokAlias',
        and 'options.gateway.id'. Note: snippet length capping is performed client-side,
        as the upstream wire API rejects unrecognized fields with HTTP 400.
        """
        body: Dict[str, Any] = {"query": query, "limit": limit}
        if settings.get("search_provider"):
            body["provider"] = settings["search_provider"]
        if settings.get("byok_alias"):
            body["byokAlias"] = settings["byok_alias"]
        body["options"] = {"gateway": {"id": settings["gateway_id"]}}
        return body

    # -- search ------------------------------------------------------------
    def search(self, query: str, limit: int = DEFAULT_LIMIT) -> Dict[str, Any]:
        settings = None
        try:
            if _interrupted():
                return search_fail("Interrupted")
            settings = self._settings()
            query = (query or "").strip() if isinstance(query, str) else ""
            if not query:
                return search_fail("Search query must not be empty")
            if len(query) > MAX_QUERY_CHARS:
                logger.info("%s: query truncated from %d to %d chars", VENDOR, len(query), MAX_QUERY_CHARS)
                query = query[:MAX_QUERY_CHARS].rstrip()
            limit = clamp_limit(limit)
            result = self._search(query, limit, settings)
            if not result["success"]:
                result["error"] = redact_error(result["error"], settings["token"])
            return result
        except ConfigError as exc:
            return search_fail(f"{VENDOR}: {exc}")
        except Exception as exc:  # noqa: BLE001 — always surface the failure envelope
            detail = redact_error(str(exc), settings["token"]) if settings else type(exc).__name__
            logger.warning("%s search error: %s", VENDOR, detail)
            return search_fail(f"{VENDOR} search failed: {detail}")

    def _search(self, query: str, limit: int, settings: Dict[str, Any]) -> Dict[str, Any]:
        url = self._endpoint(settings["account_id"])
        body = self.build_body(query, limit, settings)
        headers = {
            "Authorization": f"Bearer {settings['token']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"hermes-cloudflare-web-search/{__version__}",
        }
        timeout = settings["timeout"]

        response: Optional[httpx.Response] = None
        for attempt in (1, 2):
            if _interrupted():
                return search_fail("Interrupted")
            try:
                response = httpx.post(url, json=body, headers=headers, timeout=timeout)
            except httpx.TimeoutException:
                if _interrupted():
                    return search_fail("Interrupted")
                if attempt == 1:
                    time.sleep(_RETRY_DELAY_SECONDS)
                    continue
                return search_fail(f"{VENDOR} timed out after {timeout:g}s")
            except httpx.RequestError as exc:
                if _interrupted():
                    return search_fail("Interrupted")
                if attempt == 1:
                    time.sleep(_RETRY_DELAY_SECONDS)
                    continue
                return search_fail(f"Could not reach {VENDOR}: {exc}")
            if _interrupted():
                return search_fail("Interrupted")
            if response.status_code in _RETRYABLE_STATUSES and attempt == 1:
                time.sleep(_RETRY_DELAY_SECONDS)
                continue
            break
        assert response is not None

        try:
            payload = response.json()
        except Exception:  # noqa: BLE001
            payload = None

        if response.status_code >= 400:
            return search_fail(self._http_error(response, payload, settings))

        if not isinstance(payload, dict):
            return search_fail(f"Could not parse {VENDOR} response as JSON")
        # Be tolerant of a v4 ``{"success", "result": {...}}`` envelope as well as the raw shape.
        data = payload.get("result") if isinstance(payload.get("result"), dict) else payload
        if payload.get("success") is False or payload.get("ok") is False:
            detail, _ = error_detail(payload)
            detail = self._redact_and_cap(detail, settings["token"], _MAX_ERROR_DETAIL_CHARS)
            return search_fail(f"{VENDOR} error: {detail or 'request was not successful'}")
        items = data.get("items")
        if not isinstance(items, list):
            return search_fail(f"Unexpected {VENDOR} response: missing 'items' list")

        rows = normalize_items(items, limit, settings["max_chars"])
        meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        logger.info(
            "%s: %d results (provider=%s, limit=%d, requestId=%s, latencyMs=%s)",
            VENDOR, len(rows), settings["search_provider"] or "default", limit,
            redact_error(str(meta.get("requestId")), settings["token"]),
            redact_error(str(meta.get("latencyMs")), settings["token"]),
        )
        return {"success": True, "data": {"web": rows}}

    @staticmethod
    def _redact_and_cap(text: str, token: str, limit: int) -> str:
        redacted = redact_error(text, token).strip()
        if len(redacted) > limit:
            return redacted[:limit].rstrip() + "\u2026"
        return redacted

    @classmethod
    def _http_error(cls, response: httpx.Response, payload: Any, settings: Dict[str, Any]) -> str:
        status = response.status_code
        detail, code = error_detail(payload)
        if detail:
            detail = cls._redact_and_cap(detail, settings["token"], _MAX_ERROR_DETAIL_CHARS)
        else:
            detail = cls._redact_and_cap(response.text or "", settings["token"], _MAX_RAW_ERROR_CHARS) or f"HTTP {status}"
        message = f"{VENDOR} returned HTTP {status}: {detail}"

        hint = ""
        if status in (401, 403) or code in ("9106", "9109", "10000", "10001"):
            hint = f"check {TOKEN_ENV} / {ACCOUNT_ENV}; {TOKEN_HELP}"
        elif code == "2001":
            hint = (
                f"AI Gateway '{settings['gateway_id']}' was not found or is not set up; create it in the "
                f"Cloudflare dashboard or set {GATEWAY_ENV}"
            )
        elif code == "web_search_byok_not_configured":
            hint = (
                f"no provider key with alias '{settings['byok_alias']}' is stored on gateway "
                f"'{settings['gateway_id']}' for this provider; fix {BYOK_ENV} or unset it to use AI Gateway credits"
            )
        elif status == 429:
            hint = "rate limited by Cloudflare; retry later"
        elif status == 402 or "credit" in detail.lower():
            hint = "load AI Gateway credits or configure a provider key (BYOK) on the gateway"
        return f"{message} ({hint})" if hint else message
