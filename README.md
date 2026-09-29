# Local Inference Chip

A Hermes plugin that shows the health of your self-hosted model server (llama.cpp, vLLM, SGLang) in the Desktop status bar, and as a `/inference` command everywhere else.

```
● LOCAL · 30 TOK/S
```

Click the chip for the card:

- **Decode tok/s**: live while generating, then "last run", then lifetime
- **Prefill tok/s**, **TTFT** (vLLM, SGLang)
- **Draft acceptance** for speculative decoding (MTP, EAGLE, draft models)
- **Prompt cache hit rate**
- **KV cache fill** (vLLM, SGLang) or **context fill** (llama.cpp, from `/slots`)
- **Queue**: running and waiting requests
- **GPU** temperature, power, and utilization (only when the server runs on the same machine)
- **Baseline + drift**: save a model's speed once, then see "18% below baseline"

The chip turns amber when decode drops below the baseline, the cache is nearly full, requests are queueing, or draft acceptance falls more than 15 points below the baseline. It sends one desktop notification per state change (down, back up, warn), never one per poll.

Zero model tokens. The plugin registers no tools and no hooks on the model path.

## Install

```bash
hermes plugins install <git-url>/local-inference-chip
```

Then add it to `plugins.enabled` in `config.yaml` (the backend is gated there), and enable **Local Inference** under Capabilities → Plugins in Desktop.

## Server requirements

| Engine | What it needs |
|---|---|
| llama.cpp | `llama-server --metrics --slots` |
| vLLM | nothing, `/metrics` is on by default |
| SGLang | `--enable-metrics` |

## Configuration

With no config, every provider in `providers:` / `custom_providers:` whose base URL is on localhost, a private network, or Tailscale is picked up automatically, and its `key_env` token is used. Configured servers that have never answered are listed as "not reachable" rather than raising alarms.

To pick servers explicitly:

```yaml
plugins:
  entries:
    local-inference-chip:
      settings:
        endpoints:
          - name: GX10
            url: http://127.0.0.1:8002
            key_env: MY_SERVER_KEY   # optional bearer token, read from .env
            engine: auto             # auto | llama.cpp | vllm | sglang
            gpu: auto                # auto (local only) | on | off
        thresholds:
          drift_pct: 15
          kv_pct: 90
          queue: 1
```

## Commands

```
/inference                    health of every server
/inference baseline [server]  save current decode speed as this model's baseline
/inference clear [server]     forget it
```

Baselines are stored per server and model in `<hermes home>/plugin-data/local-inference-chip/baselines.json`.

## How the numbers are measured

- **llama.cpp** only adds a request's tokens to `/metrics` once the request completes. While a request is still generating, the chip reads each slot's decoded-token count from `/slots` and divides by wall time (labelled "in flight"). When the request completes, it switches to llama.cpp's own decode timer (labelled "live"), which matches the `predicted_per_second` the server reports.
- **vLLM, SGLang**: per-stream decode speed comes from mean inter-token latency; aggregate throughput is the fallback.
- A counter reset (server restart) gives no rate for that window instead of a negative number.

## Disclosure

The plugin reads `/metrics`, `/slots` and `/v1/models` from the servers you configured, and sends each provider's own bearer token only to that same endpoint. It runs `nvidia-smi` only when the server is on this machine. It makes no other network calls.
