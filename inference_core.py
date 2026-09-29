"""Engine-agnostic health readout for self-hosted inference servers.

Pure stdlib. Reads a server's Prometheus ``/metrics`` (llama.cpp ``llama-server --metrics``,
vLLM, SGLang), optionally ``/slots`` (llama.cpp context fill) and ``nvidia-smi`` (only when the
server is on this machine), and turns two successive samples into live rates. No model tokens,
no network beyond the configured endpoint.
"""
from __future__ import annotations

import ipaddress
import json
import math
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

PLUGIN_ID = "local-inference-chip"

# ---------------------------------------------------------------- Prometheus text parsing

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([^\s]+)")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

Metrics = Dict[str, List[Tuple[Dict[str, str], float]]]


def parse_prometheus(text: str) -> Metrics:
    """``{name: [(labels, value), ...]}``; comments, blanks and unparsable values are skipped."""
    out: Metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        labels = dict(_LABEL.findall(m.group(2) or ""))
        out.setdefault(m.group(1), []).append((labels, value))
    return out


def _total(metrics: Metrics, *names: str) -> Optional[float]:
    """Sum of every series of the first name present (engines rename across versions)."""
    for name in names:
        series = metrics.get(name)
        if series:
            vals = [v for _, v in series if not math.isnan(v)]
            if vals:
                return sum(vals)
    return None


def _max(metrics: Metrics, *names: str) -> Optional[float]:
    for name in names:
        series = metrics.get(name)
        if series:
            vals = [v for _, v in series if not math.isnan(v)]
            if vals:
                return max(vals)
    return None


# ---------------------------------------------------------------- engine mapping tables
# Each field lists candidate metric names in preference order. Counters (``*_total``) become
# rates via deltas between two samples; gauges are read as-is. Keep this table the only place
# that knows metric names.

ENGINES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "llama.cpp": {
        "gen_tokens": ("llamacpp:tokens_predicted_total",),
        "gen_seconds": ("llamacpp:tokens_predicted_seconds_total",),
        "prompt_tokens": ("llamacpp:prompt_tokens_total",),
        "prompt_seconds": ("llamacpp:prompt_seconds_total",),
        "cached_prompt_tokens": ("llamacpp:prompt_tokens_cached_total",),
        "draft_tokens": ("llamacpp:spec_decode_num_draft_tokens_total",),
        "accepted_tokens": ("llamacpp:spec_decode_num_accepted_tokens_total",),
        "running": ("llamacpp:requests_processing",),
        "waiting": ("llamacpp:requests_deferred",),
    },
    "vllm": {
        "gen_tokens": ("vllm:generation_tokens_total", "vllm:generation_tokens"),
        "prompt_tokens": ("vllm:prompt_tokens_total", "vllm:prompt_tokens"),
        "cached_prompt_tokens": ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"),
        "cache_queries": ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"),
        "ttft_sum": ("vllm:time_to_first_token_seconds_sum",),
        "ttft_count": ("vllm:time_to_first_token_seconds_count",),
        "itl_sum": ("vllm:inter_token_latency_seconds_sum", "vllm:time_per_output_token_seconds_sum"),
        "itl_count": ("vllm:inter_token_latency_seconds_count", "vllm:time_per_output_token_seconds_count"),
        "draft_tokens": ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_draft_tokens"),
        "accepted_tokens": ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens"),
        "running": ("vllm:num_requests_running",),
        "waiting": ("vllm:num_requests_waiting",),
        "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    },
    "sglang": {
        "gen_tokens": ("sglang:generation_tokens_total",),
        "prompt_tokens": ("sglang:prompt_tokens_total",),
        "cached_prompt_tokens": ("sglang:cached_tokens_total",),
        "ttft_sum": ("sglang:time_to_first_token_seconds_sum",),
        "ttft_count": ("sglang:time_to_first_token_seconds_count",),
        "itl_sum": ("sglang:inter_token_latency_seconds_sum",),
        "itl_count": ("sglang:inter_token_latency_seconds_count",),
        "running": ("sglang:num_running_reqs",),
        "waiting": ("sglang:num_queue_reqs",),
        "kv_usage": ("sglang:token_usage",),
        "spec_accept_rate": ("sglang:spec_accept_rate",),
        "gen_throughput": ("sglang:gen_throughput",),
    },
}

_GAUGES = {"running", "waiting", "kv_usage", "spec_accept_rate", "gen_throughput"}


def detect_engine(metrics: Metrics) -> Optional[str]:
    prefixes = {name.split(":", 1)[0] for name in metrics if ":" in name}
    for engine, prefix in (("llama.cpp", "llamacpp"), ("vllm", "vllm"), ("sglang", "sglang")):
        if prefix in prefixes:
            return engine
    return None


def extract(metrics: Metrics, engine: str) -> Dict[str, Optional[float]]:
    table = ENGINES[engine]
    return {key: (_max if key in _GAUGES else _total)(metrics, *names) for key, names in table.items()}


# ---------------------------------------------------------------- derived readout

def _delta(cur: Dict[str, Optional[float]], prev: Dict[str, Optional[float]], key: str) -> Optional[float]:
    a, b = cur.get(key), prev.get(key)
    if a is None or b is None:
        return None
    d = a - b
    return d if d >= 0 else None  # counter reset (server restart): no rate this window


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den <= 0:
        return None
    return num / den


def derive(engine: str, cur: Dict[str, Optional[float]], prev: Optional[Dict[str, Optional[float]]],
           wall_dt: Optional[float]) -> Dict[str, Any]:
    """Live window (cur - prev) plus lifetime figures. ``None`` means the engine does not expose it."""
    live: Dict[str, Optional[float]] = {k: None for k in (
        "decode_tps", "prefill_tps", "ttft_s", "spec_accept", "cache_hit", "gen_tokens")}
    if prev is not None and wall_dt and wall_dt > 0:
        gen = _delta(cur, prev, "gen_tokens")
        live["gen_tokens"] = gen
        if engine == "llama.cpp":
            # llama.cpp times decode itself: tokens / seconds spent generating = true decode speed.
            live["decode_tps"] = _ratio(gen, _delta(cur, prev, "gen_seconds")) if gen else None
            live["prefill_tps"] = _ratio(_delta(cur, prev, "prompt_tokens"), _delta(cur, prev, "prompt_seconds"))
        else:
            itl = _ratio(_delta(cur, prev, "itl_sum"), _delta(cur, prev, "itl_count"))
            if itl:  # per-stream decode speed from mean inter-token latency
                live["decode_tps"] = 1.0 / itl
            elif gen:
                live["decode_tps"] = gen / wall_dt  # aggregate throughput fallback
            live["ttft_s"] = _ratio(_delta(cur, prev, "ttft_sum"), _delta(cur, prev, "ttft_count"))
        draft = _delta(cur, prev, "draft_tokens")
        live["spec_accept"] = _ratio(_delta(cur, prev, "accepted_tokens"), draft) if draft else None
        cached, prompt = _delta(cur, prev, "cached_prompt_tokens"), _delta(cur, prev, "prompt_tokens")
        if engine == "vllm":
            live["cache_hit"] = _ratio(cached, _delta(cur, prev, "cache_queries"))
        elif cached is not None and prompt is not None and cached + prompt > 0:
            live["cache_hit"] = cached / (cached + prompt)
    if engine == "sglang" and cur.get("spec_accept_rate") is not None and live["spec_accept"] is None:
        live["spec_accept"] = cur["spec_accept_rate"]
    lifetime = {
        "decode_tps": _ratio(cur.get("gen_tokens"), cur.get("gen_seconds")),
        "spec_accept": _ratio(cur.get("accepted_tokens"), cur.get("draft_tokens")),
        "gen_tokens": cur.get("gen_tokens"),
    }
    return {
        "live": live,
        "lifetime": lifetime,
        "running": cur.get("running"),
        "waiting": cur.get("waiting"),
        "kv_usage": cur.get("kv_usage"),
    }


def _slot_decoded(slot: Dict[str, Any]) -> int:
    return sum((nt.get("n_decoded") or 0) for nt in (slot.get("next_token") or []) if isinstance(nt, dict))


def llamacpp_inflight(slots: Any) -> Dict[Any, Tuple[Any, int]]:
    """``{slot_id: (task_id, n_decoded)}`` for slots currently generating."""
    out: Dict[Any, Tuple[Any, int]] = {}
    if isinstance(slots, list):
        for s in slots:
            if isinstance(s, dict) and s.get("is_processing"):
                out[s.get("id")] = (s.get("id_task"), _slot_decoded(s))
    return out


def inflight_rate(prev: Optional[Dict[Any, Tuple[Any, int]]], cur: Dict[Any, Tuple[Any, int]],
                  dt: Optional[float]) -> Optional[float]:
    """Aggregate tok/s across slots that stayed on the same task between two samples.

    llama.cpp only folds tokens into /metrics counters when a request completes, so this is the
    only honest mid-generation reading. It is wall-clock (includes sampling and HTTP overhead)."""
    if not prev or not cur or not dt or dt <= 0:
        return None
    gained, matched = 0, False
    for sid, (task, n) in cur.items():
        p = prev.get(sid)
        if p and p[0] == task and n >= p[1]:
            gained += n - p[1]
            matched = True
    return gained / dt if matched and gained > 0 else None


def llamacpp_context_fill(slots: Any) -> Optional[Dict[str, Any]]:
    """Largest slot fill from llama.cpp ``/slots`` (it has no KV-usage gauge)."""
    if not isinstance(slots, list) or not slots:
        return None
    best = None
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        n_ctx = slot.get("n_ctx") or 0
        # n_prompt_tokens already grows with decoded tokens on current llama-server builds, so take
        # the larger of the two rather than summing (summing double-counts the reply).
        used = max(slot.get("n_prompt_tokens") or 0, _slot_decoded(slot))
        if n_ctx > 0:
            fill = min(used / n_ctx, 1.0)
            if best is None or fill > best["fill"]:
                best = {"fill": fill, "tokens": used, "n_ctx": n_ctx,
                        "processing": bool(slot.get("is_processing"))}
    return best


# ---------------------------------------------------------------- GPU (local only)

_SMI_FIELDS = "name,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total"


def _num(s: str) -> Optional[float]:
    s = s.strip().split(" ")[0]
    try:
        return float(s)
    except ValueError:
        return None  # "[N/A]" on unified-memory parts (GB10)


def read_gpu(runner: Callable[[List[str]], str] | None = None) -> Optional[List[Dict[str, Any]]]:
    run = runner
    if run is None:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return None

        def run(args: List[str]) -> str:
            return subprocess.run([exe, *args], capture_output=True, text=True, timeout=4, check=True).stdout
    try:
        raw = run([f"--query-gpu={_SMI_FIELDS}", "--format=csv,noheader,nounits"])
    except Exception:
        return None
    gpus = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        gpus.append({"name": parts[0], "temp_c": _num(parts[1]), "power_w": _num(parts[2]),
                     "util_pct": _num(parts[3]), "mem_used_mib": _num(parts[4]),
                     "mem_total_mib": _num(parts[5])})
    return gpus or None


def is_local_host(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_private_host(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    if host == "localhost" or host.endswith(".local") or host.endswith(".ts.net"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip in ipaddress.ip_network("100.64.0.0/10")


# ---------------------------------------------------------------- endpoints

@dataclass
class Endpoint:
    name: str
    url: str                      # server root, no trailing /v1
    key_env: Optional[str] = None
    engine: str = "auto"
    gpu: str = "auto"             # auto = nvidia-smi only when the server is on this machine
    auto: bool = False            # discovered from providers (not explicitly configured)

    @property
    def id(self) -> str:
        return re.sub(r"[^a-z0-9_-]+", "-", self.name.lower()).strip("-") or "server"


def _root(url: str) -> str:
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def endpoints_from_config(cfg: Dict[str, Any]) -> List[Endpoint]:
    """Explicit ``plugins.entries.<id>.settings.endpoints`` wins; otherwise every configured
    provider whose base URL is loopback/private (self-hosted) is a candidate."""
    settings = (((cfg.get("plugins") or {}).get("entries") or {}).get(PLUGIN_ID) or {}).get("settings") or {}
    explicit = settings.get("endpoints")
    if isinstance(explicit, list) and explicit:
        out = []
        for e in explicit:
            if isinstance(e, dict) and e.get("url"):
                out.append(Endpoint(name=str(e.get("name") or e["url"]), url=_root(str(e["url"])),
                                    key_env=e.get("key_env"), engine=str(e.get("engine") or "auto"),
                                    gpu=str(e.get("gpu") or "auto")))
        return out
    found: List[Endpoint] = []
    seen = set()
    for name, p in ((cfg.get("providers") or {}).items() if isinstance(cfg.get("providers"), dict) else []):
        url = isinstance(p, dict) and (p.get("api") or p.get("base_url"))
        if url and is_private_host(url) and _root(url) not in seen:
            seen.add(_root(url))
            found.append(Endpoint(name=str(p.get("name") or name), url=_root(url), key_env=p.get("key_env"),
                                  auto=True))
    for p in cfg.get("custom_providers") or []:
        url = isinstance(p, dict) and (p.get("base_url") or p.get("api"))
        if url and is_private_host(url) and _root(url) not in seen:
            seen.add(_root(url))
            found.append(Endpoint(name=str(p.get("name") or url), url=_root(url), key_env=p.get("key_env"),
                                  auto=True))
    return found


def thresholds_from_config(cfg: Dict[str, Any]) -> Dict[str, float]:
    settings = (((cfg.get("plugins") or {}).get("entries") or {}).get(PLUGIN_ID) or {}).get("settings") or {}
    t = settings.get("thresholds") or {}
    return {
        "drift_pct": float(t.get("drift_pct", 15)),     # amber when decode is this % under baseline
        "kv_pct": float(t.get("kv_pct", 90)),           # amber when KV/context fill passes this
        "queue": float(t.get("queue", 1)),              # amber when this many requests wait
    }


# ---------------------------------------------------------------- sampler

Fetch = Callable[[str, Optional[str], float], Tuple[int, str]]


def http_get(url: str, token: Optional[str], timeout: float = 3.0) -> Tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-local-inference-chip"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(4_000_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""


@dataclass
class _State:
    values: Optional[Dict[str, Optional[float]]] = None
    at: Optional[float] = None
    model: Optional[str] = None
    model_at: float = 0.0
    last_active: Optional[Dict[str, Any]] = None  # most recent window that generated tokens
    last_result: Optional[Dict[str, Any]] = None
    last_result_at: float = 0.0
    ever_up: bool = False
    inflight: Optional[Dict[Any, Tuple[Any, int]]] = None


class Sampler:
    """Keeps the previous sample per endpoint so each call returns a live window.

    Calls closer than ``min_window`` seconds reuse the last result (several desktop windows
    polling must not shrink the window to noise)."""

    def __init__(self, fetch: Fetch = http_get, gpu_reader: Callable[[], Any] = read_gpu,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 min_window: float = 2.0):
        self._fetch, self._gpu, self._clock, self._wall = fetch, gpu_reader, clock, wall
        self._min_window = min_window
        self._states: Dict[str, _State] = {}
        self._lock = threading.Lock()

    def sample(self, ep: Endpoint, token: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            st = self._states.setdefault(f"{ep.id}|{ep.url}", _State())
            now = self._clock()
            if st.last_result is not None and now - st.last_result_at < self._min_window:
                return st.last_result
            result = self._sample(ep, token, st, now)
            st.ever_up = st.ever_up or result.get("status") == "up"
            if result.get("status") == "down" and ep.auto and not st.ever_up:
                # A discovered provider that never answered is a stale config entry, not an outage.
                result = {**result, "status": "offline"}
            result["auto"] = ep.auto
            st.last_result, st.last_result_at = result, now
            return result

    def _sample(self, ep: Endpoint, token: Optional[str], st: _State, now: float) -> Dict[str, Any]:
        base = {"id": ep.id, "name": ep.name, "url": ep.url, "checked_at": self._wall()}
        try:
            status, body = self._fetch(f"{ep.url}/metrics", token, 3.0)
        except Exception as e:  # connection refused, timeout, DNS
            st.values, st.at = None, None
            return {**base, "status": "down", "error": type(e).__name__}
        if status in (401, 403):
            return {**base, "status": "auth", "error": f"HTTP {status} (set key_env)"}
        if status == 404 or (status == 200 and not body.strip()):
            return {**base, "status": "no-metrics",
                    "error": "no /metrics (llama-server needs --metrics)"}
        if status != 200:
            return {**base, "status": "down", "error": f"HTTP {status}"}
        metrics = parse_prometheus(body)
        engine = ep.engine if ep.engine in ENGINES else detect_engine(metrics)
        if engine is None:
            return {**base, "status": "no-metrics", "error": "unrecognised metrics format"}
        values = extract(metrics, engine)
        prev, dt = st.values, (now - st.at) if st.at is not None else None
        d = derive(engine, values, prev, dt)
        st.values, st.at = values, now
        ctx_fill = None
        d["live"]["decode_source"] = "metrics" if d["live"].get("decode_tps") else None
        if engine == "llama.cpp":
            slots = None
            try:
                s_status, s_body = self._fetch(f"{ep.url}/slots", token, 2.0)
                if s_status == 200:
                    slots = json.loads(s_body)
            except Exception:
                slots = None
            if slots is not None:
                ctx_fill = llamacpp_context_fill(slots)
                cur_inflight = llamacpp_inflight(slots)
                rate = inflight_rate(st.inflight, cur_inflight, dt)
                st.inflight = cur_inflight
                if rate and not d["live"].get("decode_tps"):
                    d["live"]["decode_tps"], d["live"]["decode_source"] = rate, "slots"
        if st.model is None or now - st.model_at > 60:
            st.model, st.model_at = self._model(ep, token) or st.model, now
        if d["live"].get("decode_tps"):
            st.last_active = {**d["live"], "at": self._wall()}
        gpu = None
        if ep.gpu == "on" or (ep.gpu == "auto" and is_local_host(ep.url)):
            gpu = self._gpu()
        return {**base, "status": "up", "engine": engine, "model": st.model,
                "window_s": dt, **d, "context": ctx_fill, "last_active": st.last_active, "gpu": gpu}

    def _model(self, ep: Endpoint, token: Optional[str]) -> Optional[str]:
        try:
            status, body = self._fetch(f"{ep.url}/v1/models", token, 3.0)
            if status == 200:
                data = json.loads(body).get("data") or []
                if data and isinstance(data[0], dict):
                    return data[0].get("id")
        except Exception:
            pass
        return None


# ---------------------------------------------------------------- health verdict + baselines

def fill_pct(snap: Dict[str, Any]) -> Optional[float]:
    if snap.get("kv_usage") is not None:
        return snap["kv_usage"] * 100
    ctx = snap.get("context")
    return ctx["fill"] * 100 if ctx else None


def verdict(snap: Dict[str, Any], baseline: Optional[Dict[str, Any]], th: Dict[str, float]) -> Dict[str, Any]:
    """``tone``: good | warn | bad | muted, plus the reasons and baseline drift."""
    if snap.get("status") != "up":
        return {"tone": "bad" if snap.get("status") == "down" else "muted",
                "reasons": [snap.get("error") or snap.get("status")], "drift_pct": None}
    reasons, drift = [], None
    live = snap["live"]
    decode = live.get("decode_tps")
    if baseline and baseline.get("decode_tps") and decode:
        drift = (decode - baseline["decode_tps"]) / baseline["decode_tps"] * 100
        if drift <= -th["drift_pct"]:
            reasons.append(f"decode {abs(drift):.0f}% below baseline")
    fp = fill_pct(snap)
    if fp is not None and fp >= th["kv_pct"]:
        reasons.append(f"cache {fp:.0f}% full")
    if snap.get("waiting") is not None and snap["waiting"] >= th["queue"]:
        reasons.append(f"{int(snap['waiting'])} queued")
    if baseline and baseline.get("spec_accept") and live.get("spec_accept") is not None:
        if live["spec_accept"] < baseline["spec_accept"] - 0.15:
            reasons.append(f"draft acceptance {live['spec_accept']*100:.0f}% "
                           f"(baseline {baseline['spec_accept']*100:.0f}%)")
    return {"tone": "warn" if reasons else "good", "reasons": reasons, "drift_pct": drift}


def baseline_key(snap: Dict[str, Any]) -> str:
    return f"{snap.get('id')}::{snap.get('model') or 'unknown'}"


class BaselineStore:
    def __init__(self, path):
        self.path = path

    def load(self) -> Dict[str, Any]:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.load().get(key)

    def save(self, key: str, snap: Dict[str, Any]) -> Dict[str, Any]:
        """Baseline = live window if it generated, else last active window, else lifetime."""
        src = snap["live"] if snap["live"].get("decode_tps") else (snap.get("last_active") or {})
        decode = src.get("decode_tps") or snap["lifetime"].get("decode_tps")
        if not decode:
            raise ValueError("no decode measurement yet: run a generation first")
        entry = {"decode_tps": decode,
                 "spec_accept": src.get("spec_accept") or snap["lifetime"].get("spec_accept"),
                 "ttft_s": src.get("ttft_s"), "model": snap.get("model"), "saved_at": time.time(),
                 "source": "live" if src is snap["live"] else ("last_active" if src else "lifetime")}
        data = self.load()
        data[key] = entry
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)
        return entry

    def clear(self, key: str) -> bool:
        data = self.load()
        if data.pop(key, None) is None:
            return False
        self.path.write_text(json.dumps(data, indent=2))
        return True


# ---------------------------------------------------------------- text rendering (slash command)

def _f(v: Optional[float], fmt: str, suffix: str = "") -> str:
    return "n/a" if v is None else f"{v:{fmt}}{suffix}"


def render_text(snap: Dict[str, Any], v: Dict[str, Any], baseline: Optional[Dict[str, Any]]) -> str:
    head = f"{snap['name']} ({snap['url']})"
    if snap.get("status") != "up":
        return f"{head}: {snap.get('status', '').upper()} — {snap.get('error')}"
    live, life = snap["live"], snap["lifetime"]
    lines = [f"{head} · {snap['engine']} · {snap.get('model') or 'model?'} · {v['tone'].upper()}"]
    if live.get("decode_tps"):
        src = "in flight" if live.get("decode_source") == "slots" else "completed"
        lines.append(f"  decode   {live['decode_tps']:.1f} tok/s (live, {src}, {snap['window_s']:.0f}s window)")
    elif snap.get("last_active"):
        la = snap["last_active"]
        lines.append(f"  decode   idle · last {_f(la.get('decode_tps'), '.1f', ' tok/s')} "
                     f"{max(0, time.time() - la['at']):.0f}s ago")
    else:
        lines.append(f"  decode   idle · lifetime {_f(life.get('decode_tps'), '.1f', ' tok/s')}")
    if live.get("prefill_tps"):
        lines.append(f"  prefill  {live['prefill_tps']:.0f} tok/s")
    if live.get("ttft_s") is not None:
        lines.append(f"  ttft     {live['ttft_s']*1000:.0f} ms")
    acc = live.get("spec_accept") if live.get("spec_accept") is not None else life.get("spec_accept")
    if acc is not None:
        tag = "live" if live.get("spec_accept") is not None else "lifetime"
        lines.append(f"  draft    {acc*100:.1f}% accepted ({tag})")
    fp = fill_pct(snap)
    if fp is not None:
        label = "kv cache" if snap.get("kv_usage") is not None else "context"
        lines.append(f"  {label:<8} {fp:.0f}% full")
    lines.append(f"  queue    {_f(snap.get('running'), '.0f')} running · {_f(snap.get('waiting'), '.0f')} waiting")
    for g in snap.get("gpu") or []:
        lines.append(f"  gpu      {g['name']} {_f(g['temp_c'], '.0f', '°C')} {_f(g['power_w'], '.0f', ' W')} "
                     f"{_f(g['util_pct'], '.0f', '% util')}")
    if baseline:
        drift = v.get("drift_pct")
        lines.append(f"  baseline {baseline['decode_tps']:.1f} tok/s"
                     + (f" · drift {drift:+.0f}%" if drift is not None else ""))
    for r in v["reasons"]:
        lines.append(f"  ! {r}")
    return "\n".join(lines)
