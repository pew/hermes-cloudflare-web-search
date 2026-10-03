"""Cloudflare Web Search backend plugin for Hermes.

Registers the ``cloudflare`` web search provider. Select it with
``web.search_backend: cloudflare`` (or ``hermes tools`` → Web Search & Extract).
"""

from __future__ import annotations

from .provider import CloudflareWebSearchProvider


def register(ctx) -> None:
    """Plugin entry point — called once at load time."""
    ctx.register_web_search_provider(CloudflareWebSearchProvider())
