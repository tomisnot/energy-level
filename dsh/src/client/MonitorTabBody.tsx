/**
 * 机甲监控 —— 右栏 tab body（React，自包含内联样式）。
 *
 * 数据源 = 权威进程的**只读监控端点**（EL `run_authority` 起 :8767）：
 *  - `GET /activity?since_seq=N` → { ok, mode, events:[{kind,actor,target,value,seq,call_id}] }
 * 每 3s 增量轮询（记住已见最大 seq，只拉更新的），新事件置顶、封顶 200 条。
 * 顶部徽标实时显示写权模式（human=绿 / ai=红 / locked=黄）。
 *
 * 连不上（权威没起/已交接）→ 如实显示离线横幅，不假绿。
 */
import { useEffect, useRef, useState } from 'react'

/** 权威只读监控端点（EL run_authority 起，固定 :8767；与 v2_bind 的默认端口同源）。 */
const MONITOR_URL = 'http://127.0.0.1:8767'
const POLL_MS = 3000
const MAX_ROWS = 200

interface ActivityEvent {
  kind: string
  actor: string
  target: string
  value: unknown
  seq: number
  call_id?: string
}

const MODE_LABEL: Record<string, string> = {
  human: '写权在人', ai: '写权在 AI', locked: '全员禁写',
}
const MODE_COLOR: Record<string, string> = {
  human: '#1a7f37', ai: '#b3261e', locked: '#9a6700',
}
const MODE_BG: Record<string, string> = {
  human: '#dafbe1', ai: '#ffebe9', locked: '#fff8c5',
}

function fmtValue(v: unknown): string {
  try {
    const s = typeof v === 'string' ? v : JSON.stringify(v)
    return s && s.length > 60 ? `${s.slice(0, 60)}…` : String(s ?? '')
  } catch { return String(v) }
}

export function MonitorTabBody() {
  const [events, setEvents] = useState<ActivityEvent[]>([])
  const [mode, setMode] = useState<string>('')
  const [offline, setOffline] = useState(false)
  const nextSince = useRef(0)

  useEffect(() => {
    let dead = false
    const poll = async () => {
      try {
        const r = await fetch(`${MONITOR_URL}/activity?since_seq=${nextSince.current}`)
        const j = await r.json()
        if (dead) return
        setOffline(false)
        setMode(j.mode ?? '')
        const fresh: ActivityEvent[] = j.events ?? []
        if (fresh.length) {
          nextSince.current = Math.max(...fresh.map(e => e.seq)) + 1
          setEvents(prev => [...fresh.slice().reverse(), ...prev].slice(0, MAX_ROWS))
        }
      } catch {
        if (!dead) setOffline(true)
      }
    }
    poll()
    const t = setInterval(poll, POLL_MS)
    return () => { dead = true; clearInterval(t) }
  }, [])

  return (
    <div style={{ height: '100%', overflow: 'auto', fontSize: 12,
      fontFamily: 'ui-monospace,SFMono-Regular,Menlo,Consolas,monospace', padding: 8 }}>
      <div style={{ position: 'sticky', top: 0, display: 'flex', gap: 8, alignItems: 'center',
        padding: '4px 0 8px', background: 'transparent' }}>
        <b style={{ letterSpacing: '.05em' }}>机甲监控</b>
        {offline
          ? <span style={{ padding: '2px 8px', borderRadius: 10, background: '#ffebe9', color: '#cf222e' }}>
              权威离线（没起或已交接）</span>
          : <span style={{ padding: '2px 8px', borderRadius: 10,
              background: MODE_BG[mode] ?? '#eee', color: MODE_COLOR[mode] ?? '#333' }}>
              {MODE_LABEL[mode] ?? (mode || '连接中…')}</span>}
      </div>
      {events.length === 0 && !offline
        ? <div style={{ opacity: .6, padding: 8 }}>暂无写事件（AI/人每改一个键都会实时出现在这里）</div>
        : events.map(e => (
          <div key={e.seq} style={{ borderBottom: '1px solid rgba(128,128,128,.18)', padding: '4px 2px',
            display: 'flex', gap: 8, alignItems: 'baseline' }}>
            <span style={{ opacity: .55 }}>#{e.seq}</span>
            <span style={{ color: e.actor === 'ai' ? '#8250df' : '#0550ae', fontWeight: 600 }}>
              {e.actor}</span>
            <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis',
              whiteSpace: 'nowrap' }} title={`${e.target} = ${fmtValue(e.value)}`}>
              {e.target} = {fmtValue(e.value)}
            </span>
          </div>
        ))}
    </div>
  )
}
