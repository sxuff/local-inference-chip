"""Behavior tests for inference_core (pure stdlib, no Hermes needed)."""
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("ic", Path(__file__).parents[1] / "inference_core.py")
ic = importlib.util.module_from_spec(_spec)
import sys
sys.modules["ic"] = ic
_spec.loader.exec_module(ic)

# Recorded from a real llama-server (GX10, --metrics --slots, MTP speculative decoding).
LLAMA_A = """# HELP llamacpp:prompt_tokens_total x
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 2.06618e+06
llamacpp:prompt_tokens_cached_total 3.82082e+07
llamacpp:prompt_seconds_total 4081.01
llamacpp:tokens_predicted_total 455388
llamacpp:tokens_predicted_seconds_total 15719.8
llamacpp:spec_decode_num_draft_tokens_total 345528
llamacpp:spec_decode_num_accepted_tokens_total 278382
llamacpp:requests_processing 0
llamacpp:requests_deferred 0
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 97867
"""


def llama_b(extra_tokens=670, extra_secs=22.43, draft=400, accepted=278, prompt=29, prompt_s=0.385):
    return (LLAMA_A
            .replace("455388", str(455388 + extra_tokens))
            .replace("15719.8", str(15719.8 + extra_secs))
            .replace("345528", str(345528 + draft))
            .replace("278382", str(278382 + accepted))
            .replace("2.06618e+06", str(2066180 + prompt))
            .replace("4081.01", str(4081.01 + prompt_s)))


VLLM = """vllm:num_requests_running{model_name="m"} 2
vllm:num_requests_waiting{model_name="m"} 3
vllm:kv_cache_usage_perc{model_name="m"} 0.93
vllm:generation_tokens_total{model_name="m"} %d
vllm:prompt_tokens_total{model_name="m"} %d
vllm:prefix_cache_hits_total{model_name="m"} %d
vllm:prefix_cache_queries_total{model_name="m"} %d
vllm:time_to_first_token_seconds_sum{model_name="m"} %f
vllm:time_to_first_token_seconds_count{model_name="m"} %d
vllm:inter_token_latency_seconds_sum{model_name="m"} %f
vllm:inter_token_latency_seconds_count{model_name="m"} %d
vllm:spec_decode_num_draft_tokens_total{model_name="m"} %d
vllm:spec_decode_num_accepted_tokens_total{model_name="m"} %d
"""

SGLANG = """sglang:num_running_reqs{model_name="m"} 1
sglang:num_queue_reqs{model_name="m"} 0
sglang:token_usage{model_name="m"} 0.41
sglang:spec_accept_rate{model_name="m"} 0.72
sglang:generation_tokens_total{model_name="m"} 1000
"""


class FakeServer:
    def __init__(self, pages):
        self.pages = pages  # path -> list of (status, body); last one repeats

    def __call__(self, url, token, timeout):
        path = "/" + url.split("/", 3)[3]
        seq = self.pages.get(path)
        if seq is None:
            return 404, ""
        item = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(item, Exception):
            raise item
        return item


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def sampler(pages, clock=None):
    return ic.Sampler(fetch=FakeServer(pages), gpu_reader=lambda: [{"name": "g"}],
                      clock=clock or Clock(), wall=lambda: 1.0, min_window=0.5)


EP = ic.Endpoint(name="box", url="http://10.0.0.5:8002")
MODELS = [(200, json.dumps({"data": [{"id": "qwen"}]}))]


def test_parser_handles_labels_comments_and_exponents():
    m = ic.parse_prometheus(LLAMA_A)
    assert m["llamacpp:prompt_tokens_total"][0][1] == 2066180.0
    assert m["llamacpp:spec_decode_num_accepted_tokens_per_pos_total"][0][0] == {"position": "0"}
    assert "llamacpp:prompt_tokens_total" in m and not any(k.startswith("#") for k in m)


@pytest.mark.parametrize("text,engine", [(LLAMA_A, "llama.cpp"), (VLLM % ((0,) * 10), "vllm"), (SGLANG, "sglang")])
def test_engine_detection(text, engine):
    assert ic.detect_engine(ic.parse_prometheus(text)) == engine


def test_llamacpp_live_window_matches_server_timings():
    clock = Clock()
    s = sampler({"/metrics": [(200, LLAMA_A), (200, llama_b())], "/v1/models": MODELS,
                 "/slots": [(200, "[]")]}, clock)
    first = s.sample(EP, None)
    assert first["status"] == "up" and first["live"]["decode_tps"] is None  # no window yet
    clock.t += 25
    snap = s.sample(EP, None)
    # 670 tokens / 22.43 s of decode, as llama-server itself reported (29.8 tok/s)
    assert snap["live"]["decode_tps"] == pytest.approx(670 / 22.43, rel=1e-3)
    assert snap["live"]["spec_accept"] == pytest.approx(278 / 400)
    assert snap["live"]["prefill_tps"] == pytest.approx(29 / 0.385, rel=1e-2)
    assert snap["model"] == "qwen"
    assert snap["gpu"] is None  # remote host: no local nvidia-smi


def test_llamacpp_in_flight_rate_from_slots():
    clock = Clock()
    slot = lambda n: json.dumps([{"id": 0, "id_task": 7, "is_processing": True, "n_ctx": 1000,
                                  "n_prompt_tokens": 100 + n, "next_token": [{"n_decoded": n}]}])
    s = sampler({"/metrics": [(200, LLAMA_A)], "/v1/models": MODELS,
                 "/slots": [(200, slot(10)), (200, slot(130))]}, clock)
    s.sample(EP, None)
    clock.t += 4
    snap = s.sample(EP, None)
    assert snap["live"]["decode_tps"] == pytest.approx(120 / 4)
    assert snap["live"]["decode_source"] == "slots"
    assert snap["context"]["tokens"] == 230  # not double-counted


def test_new_task_on_same_slot_is_not_a_rate():
    clock = Clock()
    a = json.dumps([{"id": 0, "id_task": 1, "is_processing": True, "n_ctx": 10, "next_token": [{"n_decoded": 5}]}])
    b = json.dumps([{"id": 0, "id_task": 2, "is_processing": True, "n_ctx": 10, "next_token": [{"n_decoded": 50}]}])
    s = sampler({"/metrics": [(200, LLAMA_A)], "/v1/models": MODELS, "/slots": [(200, a), (200, b)]}, clock)
    s.sample(EP, None)
    clock.t += 4
    assert s.sample(EP, None)["live"]["decode_tps"] is None


def test_counter_reset_yields_no_rate_not_negative():
    clock = Clock()
    s = sampler({"/metrics": [(200, llama_b()), (200, LLAMA_A)], "/v1/models": MODELS, "/slots": [(200, "[]")]}, clock)
    s.sample(EP, None)
    clock.t += 5
    assert s.sample(EP, None)["live"]["decode_tps"] is None


def test_vllm_per_stream_decode_ttft_cache_and_kv():
    clock = Clock()
    a = VLLM % (1000, 5000, 0, 0, 0.0, 0, 0.0, 0, 0, 0)
    b = VLLM % (1500, 6000, 600, 1000, 0.8, 4, 10.0, 500, 200, 150)
    s = sampler({"/metrics": [(200, a), (200, b)], "/v1/models": MODELS}, clock)
    s.sample(EP, None)
    clock.t += 5
    snap = s.sample(EP, None)
    assert snap["engine"] == "vllm"
    assert snap["live"]["decode_tps"] == pytest.approx(50.0)  # 1 / (10 s / 500 tokens)
    assert snap["live"]["ttft_s"] == pytest.approx(0.2)
    assert snap["live"]["cache_hit"] == pytest.approx(0.6)
    assert snap["live"]["spec_accept"] == pytest.approx(0.75)
    assert snap["kv_usage"] == pytest.approx(0.93) and snap["waiting"] == 3


def test_sglang_gauges():
    s = sampler({"/metrics": [(200, SGLANG)], "/v1/models": MODELS})
    snap = s.sample(EP, None)
    assert snap["engine"] == "sglang" and snap["kv_usage"] == pytest.approx(0.41)
    assert snap["live"]["spec_accept"] == pytest.approx(0.72)


@pytest.mark.parametrize("page,status", [
    ((401, ""), "auth"), ((404, ""), "no-metrics"), ((200, ""), "no-metrics"),
    ((500, ""), "down"), (ConnectionRefusedError(), "down"), ((200, "foo 1\n"), "no-metrics"),
])
def test_failure_states(page, status):
    assert sampler({"/metrics": [page]}).sample(EP, None)["status"] == status


def test_discovered_endpoint_never_up_is_offline_not_down():
    ep = ic.Endpoint(name="stale", url="http://10.0.0.9:1", auto=True)
    assert sampler({"/metrics": [ConnectionRefusedError()]}).sample(ep, None)["status"] == "offline"


def test_discovered_endpoint_that_was_up_then_dies_is_down():
    clock = Clock()
    ep = ic.Endpoint(name="box", url="http://10.0.0.9:1", auto=True)
    s = sampler({"/metrics": [(200, LLAMA_A), ConnectionRefusedError()], "/v1/models": MODELS, "/slots": [(200, "[]")]}, clock)
    assert s.sample(ep, None)["status"] == "up"
    clock.t += 5
    assert s.sample(ep, None)["status"] == "down"


def test_min_window_dedupes_concurrent_pollers():
    clock = Clock()
    fake = FakeServer({"/metrics": [(200, LLAMA_A)], "/v1/models": MODELS, "/slots": [(200, "[]")]})
    calls = []
    s = ic.Sampler(fetch=lambda *a: calls.append(a) or fake(*a), gpu_reader=lambda: None,
                   clock=clock, wall=lambda: 1.0, min_window=2.0)
    s.sample(EP, None)
    n = len(calls)
    s.sample(EP, None)
    assert len(calls) == n


def test_gpu_only_for_local_servers():
    local = ic.Endpoint(name="l", url="http://127.0.0.1:8080")
    s = sampler({"/metrics": [(200, LLAMA_A)], "/v1/models": MODELS, "/slots": [(200, "[]")]})
    assert s.sample(local, None)["gpu"] == [{"name": "g"}]


def test_nvidia_smi_na_fields_on_unified_memory():
    out = ic.read_gpu(lambda args: "NVIDIA GB10, 48, 11.94, 0, [N/A], [N/A]\n")
    assert out[0]["temp_c"] == 48 and out[0]["mem_used_mib"] is None


def test_discovery_keeps_only_private_providers_and_strips_v1():
    cfg = {"providers": {"gx": {"api": "http://172.17.0.1:8001/v1", "key_env": "K"},
                         "cloud": {"api": "https://api.openai.com/v1"},
                         "ts": {"api": "http://100.65.1.2:30002/v1"}},
           "custom_providers": [{"name": "loc", "base_url": "http://localhost:8080/v1"}]}
    eps = ic.endpoints_from_config(cfg)
    assert [e.url for e in eps] == ["http://172.17.0.1:8001", "http://100.65.1.2:30002", "http://localhost:8080"]
    assert eps[0].key_env == "K" and all(e.auto for e in eps)


def test_explicit_endpoints_override_discovery():
    cfg = {"providers": {"gx": {"api": "http://127.0.0.1:1/v1"}},
           "plugins": {"entries": {"local-inference-chip": {"settings": {
               "endpoints": [{"name": "Mine", "url": "http://10.1.1.1:8000/v1", "engine": "vllm"}]}}}}}
    eps = ic.endpoints_from_config(cfg)
    assert len(eps) == 1 and eps[0].engine == "vllm" and not eps[0].auto


def _up(decode=None, spec=None, kv=None, waiting=0):
    return {"status": "up", "live": {"decode_tps": decode, "spec_accept": spec}, "kv_usage": kv,
            "waiting": waiting, "context": None}


TH = {"drift_pct": 15, "kv_pct": 90, "queue": 1}


def test_verdict_drift_kv_queue_and_draft_collapse():
    base = {"decode_tps": 30.0, "spec_accept": 0.8}
    assert ic.verdict(_up(29.0, 0.78), base, TH)["tone"] == "good"
    v = ic.verdict(_up(24.0, 0.5, kv=0.95, waiting=2), base, TH)
    assert v["tone"] == "warn" and v["drift_pct"] == pytest.approx(-20)
    assert len(v["reasons"]) == 4
    assert ic.verdict({"status": "down", "error": "x"}, None, TH)["tone"] == "bad"


def test_baseline_store_roundtrip(tmp_path):
    store = ic.BaselineStore(tmp_path / "b.json")
    with pytest.raises(ValueError):
        store.save("k", {"live": {}, "lifetime": {}})
    snap = {"live": {}, "last_active": {"decode_tps": 31.0, "spec_accept": 0.7}, "lifetime": {"decode_tps": 29}, "model": "m"}
    assert store.save("k", snap)["decode_tps"] == 31.0
    assert store.get("k")["source"] == "last_active"
    assert store.clear("k") and store.get("k") is None


def test_render_text_is_readable():
    clock = Clock()
    s = sampler({"/metrics": [(200, LLAMA_A), (200, llama_b())], "/v1/models": MODELS, "/slots": [(200, "[]")]}, clock)
    s.sample(EP, None)
    clock.t += 25
    snap = s.sample(EP, None)
    text = ic.render_text(snap, ic.verdict(snap, None, TH), None)
    assert "29.9 tok/s" in text and "69.5% accepted (live)" in text
