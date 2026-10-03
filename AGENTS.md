# Repository Instructions

This repository is a standalone Hermes Agent web search provider plugin for Cloudflare's Web Search API.

## Architecture & Contract

- Manifest: `plugin.yaml` declaring `kind: backend` and `provides_web_providers: [cloudflare]`.
- Provider: Subclasses `agent.web_search_provider.WebSearchProvider`.
  - Name: `cloudflare`
  - Display name: `Cloudflare Web Search`
  - Capabilities: `supports_search() -> True`, `supports_extract() -> False`
  - Availability check: Network-free check verifying account ID and token are configured via environment or Hermes configuration.
- Wire API: Cloudflare REST endpoint `POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/websearch/`.
- Envelope: Maps `items[]` from Cloudflare response to Hermes' standard `{"success": True, "data": {"web": [{"title", "url", "description", "position"}, ...]}}`.
- Tests: Located in `tests/test_provider.py` using a lightweight HTTP server double (`_Handler` / `_Double`).

## Failure Handling

- Redact credentials from returned errors and logs before truncating diagnostics. Do not log search queries.
- Check interruption before each HTTP attempt and after responses or transport failures.
- Reject non-finite or non-positive timeouts. Report configuration read failures instead of silently using fallback settings.

## Running Tests

From the repository root, set `HERMES_SOURCE` to your Hermes Agent source checkout
and run tests using its `uv` environment:

```bash
HERMES_SOURCE=/path/to/hermes-agent
PYTHONPATH="$HERMES_SOURCE:$PWD" \
  uv run --project "$HERMES_SOURCE" python -m unittest discover -s tests -v
```

## Compilation Check

Compilation does not require a Hermes checkout or installed dependencies:

```bash
uv run --no-project python -m compileall -q __init__.py provider.py tests/test_provider.py
```
