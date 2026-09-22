/**
 * 驾驶舱右栏 **tab body**（native）：注册进 `sidebar.right.pane.tab`（key=类型 id）。
 *
 * 布局**对齐原设计**（cockpit static 的两页签）：顶部页签 [飞行记录仪 | 配置态] +
 * 告警徽标；飞行记录仪=全宽逐行折叠（折叠=概括、展开=before→after 全量）；配置态=
 * 族→前缀二级结构树（显示名去前缀）。**不并排双栏**。
 *
 * 数据：fetch cockpit HTTP 读原语 /api/history、/api/config-tree、/api/alerts。
 * 重渲染保留展开态（openSet 记 key）与当前页签。
 */
import { useEffect, useRef } from 'react'
import { readBootstrap } from './bootstrap.ts'

const TAB_CSS = `
.re0-tab-native{height:100%;overflow:auto;background:#f6f8fa;color:#1f2328;font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.re0-tab-native .re0-tabs{position:sticky;top:0;display:flex;gap:4px;align-items:center;padding:6px 8px;background:#f6f8fa;border-bottom:1px solid #d0d7de;z-index:2}
.re0-tab-native .re0-tabs button{cursor:pointer;border:1px solid #d0d7de;background:#fff;border-radius:6px;padding:3px 10px;font:inherit;color:#57606a}
.re0-tab-native .re0-tabs button.on{background:#0969da;border-color:#0969da;color:#fff}
.re0-tab-native .re0-badge{margin-left:auto;padding:2px 8px;border-radius:10px;font-size:11px}
.re0-tab-native .re0-badge.ok{background:#dafbe1;color:#1a7f37}
.re0-tab-native .re0-badge.warn{background:#fff8c5;color:#9a6700}
.re0-tab-native .re0-badge.error{background:#ffebe9;color:#cf222e}
.re0-tab-native .re0-page{padding:8px}
.re0-tab-native details.re0-ev{border:1px solid #d0d7de;border-radius:6px;background:#fff;margin:0 0 4px}
.re0-tab-native details.re0-ev>summary{cursor:pointer;padding:4px 8px;color:#1f2328;list-style:none;display:flex;gap:8px}
.re0-tab-native details.re0-ev>summary::-webkit-details-marker{display:none}
.re0-tab-native details.re0-ev>summary::before{content:'▸';color:#8b949e}
.re0-tab-native details.re0-ev[open]>summary::before{content:'▾'}
.re0-tab-native details.re0-ev .seq{color:#0969da}
.re0-tab-native details.re0-ev .actor-ai{color:#8250df}
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

export function CockpitTabBody() {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!document.querySelector('style[data-re0-tab-css]')) {
      const s = document.createElement('style')
      s.setAttribute('data-re0-tab-css', '1')
      s.textContent = TAB_CSS
      document.head.appendChild(s)
    }
    const root = ref.current
    if (!root) return
    const url = (readBootstrap()?.cockpitUrl || '').replace(/\/$/, '')
    let page: Page = 'rec'
    const openSet = new Set<string>()

    const esc = (x: any): string =>
      String(x ?? '').replace(/[&<>"]/g, (c) =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c] as string))

    const recPage = (evs: any[]): string =>
      evs.slice(-60).reverse().map((e) => {
        const k = 'ev:' + e.seq
        const open = openSet.has(k) ? ' open' : ''
        return `<details class="re0-ev" data-key="${k}"${open}><summary>` +
          `<span class="seq">#${e.seq}</span>` +
          `<span class="${e.actor === 'ai' ? 'actor-ai' : ''}">${esc(e.actor)}</span>` +
          `<span>${esc(e.kind)}</span><span>${esc(e.target)}</span></summary>` +
          `<div class="body">${esc(e.reason || '')}\nbefore: ${esc(JSON.stringify(e.before))}\n` +
          `after: ${esc(JSON.stringify(e.after))}</div></details>`
      }).join('')

    const cfgPage = (groups: any): string =>
      Object.entries(groups || {}).map(([fam, subs]: any) =>
        `<details class="re0-fam" data-key="fam:${fam}"${openSet.has('fam:' + fam) ? ' open' : ''}>` +
        `<summary>${esc(fam)}</summary>` +
        (subs || []).map((sg: any) =>
          `<details class="re0-sub" data-key="sub:${fam}/${sg.prefix}"` +
          `${openSet.has('sub:' + fam + '/' + sg.prefix) ? ' open' : ''}>` +
          `<summary>${esc(sg.prefix || '(top)')} (${(sg.rows || []).length})</summary><table>` +
          (sg.rows || []).map((r: any) =>
            `<tr><td title="${esc(r.name)}">${esc(r.short || r.name)}</td>` +
            `<td>${esc(r.value)}${r.unit ? ' <i>' + esc(r.unit) + '</i>' : ''}</td></tr>`).join('') +
          '</table></details>').join('') +
        '</details>').join('')

    const render = async (): Promise<void> => {
      if (!url) { root.innerHTML = '<div class="re0-native-msg">cockpitUrl 未注入</div>'; return }
      try {
        const [h, t, al] = await Promise.all([
          (await fetch(url + '/api/history?since_seq=0')).json(),
          (await fetch(url + '/api/config-tree')).json(),
          (await fetch(url + '/api/alerts')).json(),
        ])
        const lvl = al.level || 'ok'
        const badge = `<span class="re0-badge ${lvl}">${lvl}</span>`
        const tabs = `<div class="re0-tabs">` +
          `<button data-page="rec" class="${page === 'rec' ? 'on' : ''}">飞行记录仪</button>` +
          `<button data-page="cfg" class="${page === 'cfg' ? 'on' : ''}">配置态</button>` +
          badge + `</div>`
        const body = page === 'rec'
          ? `<div class="re0-page">${recPage(h.events || [])}</div>`
          : `<div class="re0-page"><h4>配置态</h4>${cfgPage(t.groups)}</div>`
        root.innerHTML = `<div class="re0-tab-native">${tabs}${body}</div>`
      } catch {
        root.innerHTML = '<div class="re0-native-msg">原生驾驶舱读取失败</div>'
      }
    }

    // 事件委托：页签切换 + 记录展开态（重渲染不丢）
    const onClick = (ev: Event): void => {
      const tgt = ev.target as HTMLElement
      const pg = tgt.closest?.('[data-page]')
      if (pg) { page = pg.getAttribute('data-page') as Page; render(); return }
      const sm = tgt.closest?.('summary')
      const dt = sm?.parentElement
      const key = dt?.getAttribute?.('data-key')
      if (key) { if (openSet.has(key)) openSet.delete(key); else openSet.add(key) }
    }
    root.addEventListener('click', onClick)
    render()
    const iv = setInterval(render, 3000)
    return () => { clearInterval(iv); root.removeEventListener('click', onClick) }
  }, [])

  return <div ref={ref} />
}
