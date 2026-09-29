"""Glue between Hermes (config, secrets, plugin storage) and the pure ``inference_core``."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent


def _core():
    """Load ``inference_core`` whether imported as a package (agent half) or by file path
    (``dashboard/plugin_api.py`` is imported standalone by the web server)."""
    name = "local_inference_chip_core"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, _HERE / "inference_core.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


core = _core()
_SAMPLER = core.Sampler()


def _config() -> Dict[str, Any]:
    from hermes_cli.config import load_config_readonly
    return load_config_readonly() or {}


def _secret(key_env: Optional[str]) -> Optional[str]:
    if not key_env:
        return None
    from hermes_cli.config import get_env_value
    return get_env_value(key_env)


def _baselines():
    from plugins.plugin_storage import plugin_data_dir
    return core.BaselineStore(plugin_data_dir(core.PLUGIN_ID) / "baselines.json")


def status() -> Dict[str, Any]:
    cfg = _config()
    th = core.thresholds_from_config(cfg)
    store = _baselines()
    servers: List[Dict[str, Any]] = []
    for ep in core.endpoints_from_config(cfg):
        snap = _SAMPLER.sample(ep, _secret(ep.key_env))
        base = store.get(core.baseline_key(snap)) if snap.get("status") == "up" else None
        servers.append({**snap, "baseline": base, "verdict": core.verdict(snap, base, th)})
    return {"servers": servers, "thresholds": th}


def _find(name: Optional[str]):
    cfg = _config()
    eps = core.endpoints_from_config(cfg)
    if not eps:
        return None, []
    if name:
        for ep in eps:
            if name in (ep.id, ep.name):
                return ep, eps
        return None, eps
    return eps[0], eps


def save_baseline(name: Optional[str] = None) -> Dict[str, Any]:
    ep, _ = _find(name)
    if ep is None:
        raise ValueError(f"unknown server {name!r}" if name else "no local inference server configured")
    snap = _SAMPLER.sample(ep, _secret(ep.key_env))
    if snap.get("status") != "up":
        raise ValueError(f"{ep.name} is {snap.get('status')}: {snap.get('error')}")
    return {"server": ep.id, "key": core.baseline_key(snap),
            "baseline": _baselines().save(core.baseline_key(snap), snap)}


def clear_baseline(name: Optional[str] = None) -> Dict[str, Any]:
    ep, _ = _find(name)
    if ep is None:
        raise ValueError("no such server")
    snap = _SAMPLER.sample(ep, _secret(ep.key_env))
    key = core.baseline_key(snap)
    return {"server": ep.id, "key": key, "cleared": _baselines().clear(key)}


_HELP = """/inference            health of every self-hosted server (auto-detected from providers)
/inference baseline [server]   save current decode speed as this model's baseline
/inference clear [server]      forget the baseline
Settings: plugins.entries.local-inference-chip.settings.{endpoints,thresholds} in config.yaml"""


def handle_slash(raw_args: str) -> str:
    argv = (raw_args or "").split()
    sub = argv[0] if argv else ""
    try:
        if sub in ("help", "-h", "--help"):
            return _HELP
        if sub == "baseline":
            r = save_baseline(argv[1] if len(argv) > 1 else None)
            b = r["baseline"]
            return (f"Baseline saved for {r['key']}: {b['decode_tps']:.1f} tok/s"
                    + (f", draft acceptance {b['spec_accept']*100:.1f}%" if b.get("spec_accept") else "")
                    + f" (from {b['source']} window)")
        if sub == "clear":
            r = clear_baseline(argv[1] if len(argv) > 1 else None)
            return f"Baseline {'cleared' if r['cleared'] else 'not set'} for {r['key']}"
        if sub:
            return f"Unknown subcommand {sub!r}.\n\n{_HELP}"
        data = status()
        if not data["servers"]:
            return ("No self-hosted inference server found. Add one under "
                    "plugins.entries.local-inference-chip.settings.endpoints, or configure a provider "
                    "whose base URL is on localhost / a private network.")
        shown = [s for s in data["servers"] if s.get("status") != "offline"]
        offline = [s["name"] for s in data["servers"] if s.get("status") == "offline"]
        out = "\n\n".join(core.render_text(s, s["verdict"], s.get("baseline")) for s in shown)
        if offline:
            out += ("\n\n" if out else "") + f"Not reachable (never seen up): {', '.join(offline)}"
        return out
    except ValueError as e:
        return f"inference: {e}"
