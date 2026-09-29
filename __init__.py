"""local-inference-chip: health of self-hosted inference servers (llama.cpp, vLLM, SGLang).

Agent half: a ``/inference`` slash command (works in CLI and every messaging platform).
Desktop half: ``desktop/plugin.js`` (status-bar chip + card) backed by ``dashboard/plugin_api.py``.
No tools, no hooks on the model path, zero model tokens.
"""
from __future__ import annotations

from . import service


def register(ctx) -> None:
    ctx.register_command(
        "inference",
        handler=service.handle_slash,
        description="Local inference server health: tok/s, draft acceptance, cache, queue, GPU.",
        args_hint="[baseline|clear|help]",
    )
