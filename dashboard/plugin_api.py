"""Backend routes, mounted at /api/plugins/local-inference-chip/ (desktop ``ctx.rest``)."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

_NAME = "local_inference_chip_service"
_svc = sys.modules.get(_NAME)
if _svc is None:
    _spec = importlib.util.spec_from_file_location(_NAME, Path(__file__).resolve().parent.parent / "service.py")
    _svc = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _svc
    _spec.loader.exec_module(_svc)

router = APIRouter()


@router.get("/status")
async def status():
    # Network + subprocess I/O: keep it off the event loop.
    return await asyncio.to_thread(_svc.status)


@router.post("/baseline")
async def baseline(body: dict | None = None):
    try:
        return await asyncio.to_thread(_svc.save_baseline, (body or {}).get("server"))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/baseline")
async def clear(server: str | None = None):
    try:
        return await asyncio.to_thread(_svc.clear_baseline, server)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
