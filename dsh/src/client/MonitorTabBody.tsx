/**
 * 机甲监控 —— 右栏 tab body：**v1 驾驶舱呈现原样整回**（用户裁决 2026-09-24）。
 *
 * 布局对齐 v1 CockpitTabBody（旧 dsh/src/client/CockpitTabBody.tsx，e366de8 退役）：
 * 顶部页签 [飞行记录仪 | 配置态] + 状态徽标；飞行记录仪 = 全宽逐行折叠
 * （折叠=概括 seq/actor/kind/reason，展开=before→after 全量）；配置态 =
 * 族→前缀二级结构树（显示名去前缀）。CSS 逐字照搬 .re0-tab-native 那套。
 *
 * 唯一换的是**数据源**：v1 fetch :8766 的 /api/history、/api/config-tree（黑匣子
 * 读磁盘 journal）；v2 fetch 权威进程 :8767 的只读 /history、/config（活桥投影，
 * 输出形状与 v1 逐字段对齐——v1 backend.py 的 config_tree/read_events 语义）。
 * 徽标：v1 是 alerts 级别（run 降级/失败/空演化规则在 v2 无对应事件流），换成
 * **写权模式**（human 绿 / ai 紫 / locked 黄）——同一位置、同一 CSS 类。
 *
 * 重渲染保留展开态（openSet 记 key）与当前页签（模块级，跨挂载不丢）。
 */
import { useEffect, useRef, useState, type MouseEvent } from 'react'

/** 权威只读监控端点（EL run_authority 起，固定 :8767；与 v2_bind 的默认端口同源）。 */
const MONITOR_URL = 'http://127.0.0.1:8767'
const POLL_MS = 3000

const TAB_CSS = `
.re0-tab-native{height:100%;overflow:auto;background:#f6f8fa;color:#1f2328;font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.re0-tab-native .re0-tabs{position:sticky;top:0;display:flex;gap:4px;align-items:center;padding:6px 8px;background:#f6f8fa;border-bottom:1px solid #d0d7de;z-index:2}
.re0-tab-native .re0-tabs button{cursor:pointer;border:1px solid #d0d7de;background:#fff;border-radius:6px;padding:3px 10px;font:inherit;color:#57606a}
.re0-tab-native .re0-tabs button.on{background:#0969da;border-color:#0969da;color:#fff}
.re0-tab-native .re0-badge{margin-left:auto;padding:2px 8px;border-radius:10px;font-size:11px}
.re0-tab-native .re0-badge.ok{background:#dafbe1;color:#1a7f37}
.re0-tab-native .re0-badge.warn{background:#fff8c5;color:#9a6700}
.re0-tab-native .re0-badge.error{background:#ffebe9;color:#cf222e}
.re0-tab-native .re0-badge.ai{background:#fbefff;color:#8250df}
.re0-tab-native .re0-page{padding:8px}
.re0-tab-native details.re0-ev{border:1px solid #d0d7de;border-radius:6px;background:#fff;margin:0 0 4px}
.re0-tab-native details.re0-ev>summary{cursor:pointer;padding:4px 8px;color:#1f2328;list-style:none;display:flex;gap:8px}
.re0-tab-native details.re0-ev>summary::-webkit-details-marker{display:none}
.re0-tab-native details.re0-ev>summary::before{content:'▸';color:#8b949e}
.re0-tab-native details.re0-ev[open]>summary::before{content:'▾'}
.re0-tab-native details.re0-ev .seq{color:#0969da}
.re0-tab-native details.re0-ev .actor-ai{color:#8250df}
.re0-tab-native details.re0-ev .re0-tag{color:#57606a;border:1px solid #d0d7de;border-radius:4px;padding:0 4px;font-size:11px}
.re0-tab-native details.re0-ev .re0-reason{color:#1f2328;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.re0-tab-native details.re0-ev .body i{color:#8b949e;font-style:normal;margin-right:4px}
.re0-tab-native details.re0-ev .body{padding:4px 8px 8px 22px;border-top:1px solid #eaeef2;color:#57606a;white-space:pre-wrap}
.re0-tab-native details.re0-sub{margin:2px 0 6px 8px;border-left:2px solid #d0d7de;padding-left:8px}
.re0-tab-native details.re0-fam{margin:0 0 6px}
.re0-tab-native summary{cursor:pointer;color:#57606a;font-weight:600;list-style:none}
.re0-tab-native summary::-webkit-details-marker{display:none}
.re0-tab-native summary::before{content:'▸ ';color:#0969da}
.re0-tab-native details[open]>summary::before{content:'▾ '}
.re0-tab-native table{width:100%;border-collapse:collapse}
.re0-tab-native td{border-bottom:1px solid #eaeef2;padding:2px 4px;vertical-align:top}
.re0-tab-native td:first-child{color:#0550ae}
.re0-tab-native h4{margin:0 0 6px;color:#0969da;font-size:11px;letter-spacing:.1em;text-transform:uppercase}
.re0-tab-native .re0-native-msg{padding:8px;color:#cf222e}
`

type Page = 'rec' | 'cfg'

interface Ev {
  seq: number; kind: string; actor: string; target: string
  before: unknown; after: unknown; reason?: string
}
interface Row { name: string; short: string; value: unknown; unit?: string; set?: { actor: string; seq: number } | null }
interface Sub { prefix: string; rows: Row[] }

// 展开态与页签跨挂载保留（模块级，与 v1 的实例级 openSet 同效且更稳）
const openSet = new Set<string>()
let savedPage: Page = 'rec'

const esc = (x: unknown): string =>
  String(x ?? '').replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c] as string))

// v1 fmtVal：截断/指针友好显示，不 dump 转义 JSON（防"乱码"）
const fmtVal = (v: unknown): string => {
  if (v === null || v === undefined) return '∅'
  if (typeof v === 'object') {
    const o = v as Record<string, unknown>
    if (o.$pointer) return '⟨数组 ' + ((o.shape || []) as unknown[]).join('×') + '⟩'
    if (o.$truncated) return '⟨截断 ' + o.kept + '/' + o.total + '⟩'
    const s = JSON.stringify(v)
    return s && s.length > 120 ? s.slice(0, 117) + '…' : s ?? String(v)
  }
  return String(v)
}

/** 飞行记录仪：v1 recPage 逐行同款（折叠=概括、展开=before→after 全量）。 */
function recPage(evs: Ev[]): string {
  return evs.slice(-60).reverse().map((e) => {
    const k = 'ev:' + e.seq
    const open = openSet.has(k) ? ' open' : ''
    return `<details class="re0-ev" data-key="${k}"${open}><summary>` +
      `<span class="seq">#${e.seq}</span>` +
      `<span class="${e.actor === 'ai' ? 'actor-ai' : ''}">${esc(e.actor)}</span>` +
      `<span class="re0-tag">${esc(e.kind)}</span>` +
      `<span class="re0-reason" title="${esc(e.target)}">${esc(e.reason || e.target)}</span>` +
      `</summary><div class="body">` +
      `<div><i>target</i>${esc(e.target)}</div>` +
      `<div><i>before → after</i>${esc(fmtVal(e.before))} → ${esc(fmtVal(e.after))}</div>` +
      `<div><i>reason</i>${esc(e.reason || '')}</div>` +
      `</div></details>`
  }).join('')
}

/** 配置态：v1 cfgPage 同款（族→前缀二级 details 树，显示名去前缀）。 */
function cfgPage(groups: Record<string, Sub[]>): string {
  return Object.entries(groups || {}).map(([fam, subs]) =>
    `<details class="re0-fam" data-key="fam:${fam}"${openSet.has('fam:' + fam) ? ' open' : ''}>` +
    `<summary>${esc(fam)}</summary>` +
    (subs || []).map((sg) =>
      `<details class="re0-sub" data-key="sub:${fam}/${sg.prefix}"` +
      `${openSet.has('sub:' + fam + '/' + sg.prefix) ? ' open' : ''}>` +
      `<summary>${esc(sg.prefix || '(top)')} (${(sg.rows || []).length})</summary><table>` +
      (sg.rows || []).map((r) =>
        `<tr><td title="${esc(r.name)}${r.set ? ' ← ' + esc(r.set.actor) + ' #' + r.set.seq : ''}">` +
        `${esc(r.short || r.name)}</td>` +
        `<td>${esc(fmtVal(r.value))}${r.unit ? ' <i>' + esc(r.unit) + '</i>' : ''}</td></tr>`).join('') +
      '</table></details>').join('') +
    '</details>').join('')
}

export function MonitorTabBody() {
  const ref = useRef<HTMLDivElement>(null)
  const [page, setPage] = useState<Page>(savedPage)
  const [offline, setOffline] = useState(false)
  const [mode, setMode] = useState('')
  const [events, setEvents] = useState<Ev[]>([])
  const [groups, setGroups] = useState<Record<string, Sub[]>>({})
  const [hasSchema, setHasSchema] = useState(false)
  const [orphans, setOrphans] = useState<Row[]>([])

  // 轮询：/history（v1 形状全量尾）+ /config（结构树）。3s 一次，展示刷新非正确性依赖。
  useEffect(() => {
    let dead = false
    const poll = async () => {
      try {
        const [h, c] = await Promise.all([
          fetch(`${MONITOR_URL}/history?since_seq=0`).then(r => r.json()),
          fetch(`${MONITOR_URL}/config`).then(r => r.json()),
        ])
        if (dead) return
        setOffline(false)
        setMode(h.mode ?? '')
        setEvents(h.events ?? [])
        setGroups(c.groups ?? {})
        setHasSchema(!!c.has_schema)
        setOrphans(c.orphans ?? [])
      } catch {
        if (!dead) setOffline(true)
      }
    }
    poll()
    const t = setInterval(poll, POLL_MS)
    return () => { dead = true; clearInterval(t) }
  }, [])

  // CSS 注入一次（v1 同款 data 属性防重）
  useEffect(() => {
    if (!document.querySelector('style[data-re0-tab-css]')) {
      const s = document.createElement('style')
      s.setAttribute('data-re0-tab-css', '1')
      s.textContent = TAB_CSS
      document.head.appendChild(s)
    }
  }, [])

  // 事件委托（v1 onClick 同款）：页签切换 + 记录展开态。React 委托挂在容器上，
  // 对 dangerouslySetInnerHTML 重建的子树同样生效（v1 的 native 委托等价物）。
  const onClick = (e: MouseEvent) => {
    const tgt = e.target as HTMLElement
    const pg = tgt.closest?.('[data-page]')
    if (pg) { const p = pg.getAttribute('data-page') as Page; savedPage = p; setPage(p); return }
    const sm = tgt.closest?.('summary')
    const dt = sm?.parentElement
    const key = dt?.getAttribute?.('data-key')
    if (key) { if (openSet.has(key)) openSet.delete(key); else openSet.add(key) }
  }

  const badge = offline
    ? `<span class="re0-badge error">权威离线</span>`
    : `<span class="re0-badge ${mode === 'ai' ? 'ai' : mode === 'locked' ? 'warn' : 'ok'}">` +
      `${esc(mode === 'ai' ? '写权 AI' : mode === 'locked' ? '全员禁写' : mode === 'human' ? '写权在人' : '连接中…')}</span>`
  const tabs = `<div class="re0-tabs">` +
    `<button data-page="rec" class="${page === 'rec' ? 'on' : ''}">飞行记录仪</button>` +
    `<button data-page="cfg" class="${page === 'cfg' ? 'on' : ''}">配置态</button>` +
    badge + `</div>`
  const body = page === 'rec'
    ? `<div class="re0-page">${events.length ? recPage(events) :
        `<div class="re0-native-msg" style="color:#57606a">${offline ? '连不上权威（软件没在 AI 模式跑？）' : '暂无写事件——AI/人每改一个键都会出现在这里'}</div>`}</div>`
    : `<div class="re0-page"><h4>配置态</h4>${
        hasSchema ? cfgPage(groups) :
        `<div class="re0-native-msg" style="color:#57606a">无 schema（纯内存装配？）——下方按快照直接列出现值</div>`}${
        (orphans.length ? `<h4>契约外（orphans）</h4><table>${orphans.map(r =>
          `<tr><td title="${esc(r.name)}">${esc(r.name)}</td><td>${esc(fmtVal(r.value))}</td></tr>`).join('')}</table>` : '')}</div>`

  return (
    <div ref={ref} onClick={onClick}
      // 数据变化后由 React 重渲染 innerHTML（v1 的 render() 等价物）
      dangerouslySetInnerHTML={{ __html: `<div class="re0-tab-native">${tabs}${body}</div>` }} />
  )
}
