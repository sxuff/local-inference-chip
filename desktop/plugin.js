// Local Inference Chip — status-bar health for self-hosted model servers.
// Data comes from this plugin's backend (dashboard/plugin_api.py) via ctx.rest.
import {
  Button, host, PALETTE_AREA, Popover, PopoverContent, PopoverTrigger, STATUSBAR_AREAS,
  StatusDot, useQuery, useQueryClient
} from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const ID = 'local-inference-chip'
const KEY = [ID, 'status']
const POLL_MS = 4000

const n = (v, d = 0) => (v == null || Number.isNaN(v) ? null : Number(v).toFixed(d))
const pct = v => (v == null ? null : `${Math.round(v * 100)}%`)

function fillPct(s) {
  if (s.kv_usage != null) return { label: 'KV cache', value: s.kv_usage }
  if (s.context) return { label: 'Context', value: s.context.fill }
  return null
}

function primary(servers) {
  const up = servers.filter(s => s.status === 'up')
  return up.find(s => s.live?.decode_tps) || up[0] || servers.find(s => s.status !== 'offline') || null
}

function chipText(s) {
  if (!s) return 'LOCAL · —'
  if (s.status !== 'up') return `LOCAL · ${s.status.toUpperCase()}`
  if (s.live?.decode_tps) return `LOCAL · ${n(s.live.decode_tps)} TOK/S`
  return 'LOCAL · IDLE'
}

function Row({ label, value, hint }) {
  if (value == null) return null
  return jsxs('div', {
    className: 'flex items-baseline justify-between gap-4 py-0.5',
    children: [
      jsx('span', { className: 'text-(--ui-text-tertiary)', children: label }),
      jsxs('span', {
        className: 'tabular-nums text-(--ui-text-secondary)',
        children: [value, hint ? jsx('span', { className: 'ml-1 text-(--ui-text-quaternary)', children: hint }) : null]
      })
    ]
  })
}

function ServerCard({ s, ctx }) {
  const qc = useQueryClient()
  const [busy, setBusy] = useState(false)
  const v = s.verdict || { tone: 'muted', reasons: [] }
  const live = s.live || {}
  const life = s.lifetime || {}
  const fill = fillPct(s)
  const decode = live.decode_tps
    ? { value: `${n(live.decode_tps, 1)} tok/s`, hint: live.decode_source === 'slots' ? 'in flight' : 'live' }
    : s.last_active?.decode_tps
      ? { value: `${n(s.last_active.decode_tps, 1)} tok/s`, hint: 'last run' }
      : { value: life.decode_tps ? `${n(life.decode_tps, 1)} tok/s` : null, hint: 'lifetime' }
  const accept = live.spec_accept != null ? { value: pct(live.spec_accept), hint: 'live' }
    : life.spec_accept != null ? { value: pct(life.spec_accept), hint: 'lifetime' } : null

  const act = async (method) => {
    setBusy(true)
    try {
      if (method === 'POST') await ctx.rest('/baseline', { method: 'POST', body: { server: s.id } })
      else await ctx.rest(`/baseline?server=${encodeURIComponent(s.id)}`, { method: 'DELETE' })
      host.notify({ kind: 'success', message: method === 'POST' ? `Baseline saved for ${s.model || s.name}` : 'Baseline cleared' })
      await qc.invalidateQueries({ queryKey: KEY })
    } catch (e) {
      host.notifyError(e, 'Could not update baseline')
    } finally {
      setBusy(false)
    }
  }

  return jsxs('div', {
    className: 'flex flex-col gap-1 border-b border-(--ui-stroke-secondary) pb-2 last:border-b-0',
    children: [
      jsxs('div', {
        className: 'flex items-center gap-2',
        children: [
          jsx(StatusDot, { tone: v.tone }),
          jsx('span', { className: 'font-medium text-(--ui-text-secondary)', children: s.name }),
          s.engine ? jsx('span', { className: 'text-(--ui-text-quaternary)', children: s.engine }) : null
        ]
      }),
      s.model ? jsx('div', { className: 'truncate text-(--ui-text-quaternary)', title: s.model, children: s.model }) : null,
      s.status !== 'up'
        ? jsx('div', { className: 'text-(--ui-text-tertiary)', children: s.error || s.status })
        : jsxs('div', {
          children: [
            jsx(Row, { label: 'Decode', value: decode.value, hint: decode.hint }),
            jsx(Row, { label: 'Prefill', value: live.prefill_tps ? `${n(live.prefill_tps)} tok/s` : null }),
            jsx(Row, { label: 'TTFT', value: live.ttft_s != null ? `${n(live.ttft_s * 1000)} ms` : null }),
            accept ? jsx(Row, { label: 'Draft acceptance', value: accept.value, hint: accept.hint }) : null,
            jsx(Row, { label: 'Prompt cache hit', value: pct(live.cache_hit) }),
            fill ? jsx(Row, { label: fill.label, value: pct(fill.value) }) : null,
            jsx(Row, { label: 'Queue', value: `${n(s.running) ?? '—'} running · ${n(s.waiting) ?? '—'} waiting` }),
            ...(s.gpu || []).map((g, i) => jsx(Row, {
              key: `gpu${i}`, label: g.name,
              value: [g.temp_c != null && `${n(g.temp_c)}°C`, g.power_w != null && `${n(g.power_w)} W`,
                g.util_pct != null && `${n(g.util_pct)}%`].filter(Boolean).join(' · ') || null
            })),
            s.baseline ? jsx(Row, {
              label: 'Baseline', value: `${n(s.baseline.decode_tps, 1)} tok/s`,
              hint: v.drift_pct != null ? `${v.drift_pct > 0 ? '+' : ''}${n(v.drift_pct)}%` : null
            }) : null,
            ...v.reasons.map((r, i) => jsx('div', { key: `r${i}`, className: 'text-(--ui-text-secondary)', children: `⚠ ${r}` })),
            jsxs('div', {
              className: 'mt-1 flex gap-1',
              children: [
                jsx(Button, { size: 'xs', variant: 'secondary', disabled: busy, onClick: () => void act('POST'), children: 'Save as baseline' }),
                s.baseline ? jsx(Button, { size: 'xs', variant: 'ghost', disabled: busy, onClick: () => void act('DELETE'), children: 'Clear' }) : null
              ]
            })
          ]
        })
    ]
  })
}

// One notification per state transition (down / back up / warn), never per poll.
function useTransitionAlerts(servers, ctx) {
  const prev = useRef(null)
  useEffect(() => {
    if (!servers) return
    const now = Object.fromEntries(servers.map(s => [s.id, s.status === 'up' ? (s.verdict?.tone || 'good') : s.status]))
    if (prev.current) {
      for (const s of servers) {
        const was = prev.current[s.id], is = now[s.id]
        if (!was || was === is || is === 'offline') continue
        let body = null
        if (is === 'down') body = `${s.name} is down (${s.error || 'no response'})`
        else if (was === 'down' && (is === 'good' || is === 'warn')) body = `${s.name} is back up`
        else if (is === 'warn') body = `${s.name}: ${(s.verdict?.reasons || []).join(', ')}`
        if (body) ctx.os.notify({ title: 'Local inference', body, silent: true })
      }
    }
    prev.current = now
  }, [servers, ctx])
}

function Chip({ ctx }) {
  const [open, setOpen] = useState(false)
  const q = useQuery({
    queryKey: KEY,
    queryFn: () => ctx.rest('/status'),
    refetchInterval: POLL_MS,
    refetchIntervalInBackground: true,
    retry: false
  })
  const servers = q.data?.servers
  useTransitionAlerts(servers, ctx)
  if (q.isError) {
    return jsx('span', {
      className: 'px-1.5 text-[0.6875rem] text-(--ui-text-quaternary)',
      title: 'Backend not loaded: add local-inference-chip to plugins.enabled in config.yaml',
      children: 'LOCAL · OFF'
    })
  }
  if (!servers || servers.length === 0) return null
  const p = primary(servers)
  const tone = p ? (p.status === 'up' ? p.verdict?.tone || 'good' : p.verdict?.tone || 'muted') : 'muted'
  const visible = servers.filter(s => s.status !== 'offline')
  const offline = servers.filter(s => s.status === 'offline')
  return jsxs(Popover, {
    open, onOpenChange: setOpen,
    children: [
      jsx(PopoverTrigger, {
        asChild: true,
        children: jsxs(Button, {
          variant: 'ghost', size: 'micro', 'aria-label': 'Local inference health',
          className: 'gap-1.5 text-[0.6875rem] tabular-nums tracking-wide',
          children: [jsx(StatusDot, { tone }), chipText(p)]
        })
      }),
      jsxs(PopoverContent, {
        side: 'top', align: 'end',
        className: 'flex w-80 max-w-[calc(100vw-1rem)] flex-col gap-2 text-xs',
        children: [
          ...visible.map(s => jsx(ServerCard, { key: s.id, s, ctx })),
          offline.length ? jsx('div', {
            className: 'text-(--ui-text-quaternary)',
            children: `Not reachable: ${offline.map(s => s.name).join(', ')}`
          }) : null
        ]
      })
    ]
  })
}

export default {
  id: ID,
  name: 'Local Inference',
  description: 'Status-bar health for llama.cpp, vLLM and SGLang servers: tok/s, draft acceptance, cache, queue, GPU.',
  defaultEnabled: false,
  register(ctx) {
    ctx.register({ id: 'chip', area: STATUSBAR_AREAS.right, order: 60, render: () => jsx(Chip, { ctx }) })
    ctx.register({
      id: 'baseline', area: PALETTE_AREA,
      data: {
        id: `${ID}.baseline`, label: 'Local inference: save current speed as baseline',
        keywords: ['llama', 'vllm', 'sglang', 'tok/s', 'baseline'],
        run: () => ctx.rest('/baseline', { method: 'POST', body: {} })
          .then(r => host.notify({ kind: 'success', message: `Baseline: ${Number(r.baseline.decode_tps).toFixed(1)} tok/s` }))
          .catch(e => host.notifyError(e, 'No generation measured yet'))
      }
    })
  }
}
