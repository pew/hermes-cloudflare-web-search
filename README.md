# Hermes Cloudflare Web Search Provider

A search backend for [Hermes Agent](https://hermes-agent.nousresearch.com/) using
[Cloudflare Web Search API](https://developers.cloudflare.com/web-search/how-to-use/).
Pair it with a separate extract backend for `web_extract`.

## Requirements

- Hermes Agent with its supported Python environment, which supplies `httpx`.
- A Cloudflare account ID and API token with **Account > Workers AI > Read** and
  **Account > AI Gateway > Read** permissions.
- An AI Gateway with credits or a stored search provider key.

## Setup

1. From this repository, link it into your Hermes plugin directory:

   ```bash
   mkdir -p ~/.hermes/plugins
   ln -s "$PWD" ~/.hermes/plugins/cloudflare-web-search
   ```

2. Enable the plugin and select the backend in `~/.hermes/config.yaml`.
   Merge these fields into existing sections; preserve other enabled plugins:

   ```yaml
   plugins:
     enabled:
       - cloudflare-web-search

   web:
     search_backend: cloudflare
     extract_backend: tavily # use an extract backend you have configured
     cloudflare:
       account_id: "your-cloudflare-account-id"
       gateway_id: "default"
       provider: "ceramic" # or exa / linkup
       # byok_alias: "your-stored-key-alias"
   ```

   Enable `byok_alias` only if that provider key is stored on the gateway.
   An explicit alias with no matching key fails instead of using credits.
   When omitted, Cloudflare uses a stored `default` key if available, otherwise credits.

3. Set the token in `~/.hermes/.env`, then restart Hermes:

   ```dotenv
   CLOUDFLARE_WEB_SEARCH_API_TOKEN=your-cloudflare-api-token
   ```

   For environment-based settings, omit `web.cloudflare` above and add these
   alongside the token. The enable/select fields from step 2 are still required:

   ```dotenv
   CLOUDFLARE_WEB_SEARCH_ACCOUNT_ID=your-cloudflare-account-id
   CLOUDFLARE_WEB_SEARCH_GATEWAY_ID=default
   CLOUDFLARE_WEB_SEARCH_PROVIDER=ceramic
   ```

## Configuration and troubleshooting

See [plugin.yaml](plugin.yaml) for YAML settings and [.env.example](.env.example)
for environment variables. YAML settings also work under
`plugins.entries.cloudflare-web-search.settings`; `web.cloudflare` takes precedence.
The legacy `plugins.cloudflare-web-search` section and `CLOUDFLARE_SEARCH_*`
environment aliases remain supported. The description cap alias is
`CLOUDFLARE_SEARCH_MAX_SNIPPET_LENGTH`.

Dedicated environment variables override YAML. For account ID and token, YAML
overrides shared `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` fallbacks.
Descriptions are capped locally; set `max_description_chars: 0` to disable truncation.
Timeouts must be finite and greater than zero. A failed config read is reported
without sending a search request.

If a request fails, the returned error identifies authentication, gateway, BYOK,
credit, or rate-limit problems where available. Check the token's permissions,
the selected account and gateway, and the gateway's credits or provider key.

## Testing

Run against a Hermes source checkout using its environment. Tests use a local HTTP
server with synthetic credentials, plus Hermes' provider class and manifest parser:

```bash
PYTHONPATH="/path/to/hermes-agent:$PWD" \
  uv run --project /path/to/hermes-agent python -m unittest discover -s tests -v
uv run --project /path/to/hermes-agent python -m compileall -q __init__.py provider.py tests/test_provider.py
```

These checks cover requests, response normalization, error handling, configuration,
and registration. They do not verify live Cloudflare authentication or billing.

## License

MIT
