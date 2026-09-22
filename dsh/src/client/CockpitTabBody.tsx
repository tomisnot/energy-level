/**
 * 驾驶舱右栏 **tab body**（native，opt-in）：注册进 `sidebar.right.pane.tab`（key=类型 id）。
 *
 * 正确路径（源码 seat.client.spec.tsx 验证）：右栏是 dockkit tab 体制——内容必须是
 * 一个**已注册 tab 类型的 tab**，列才展开；shadow `rightbar.session` 或 DOM-append 到
 * 可能塌掉的列都是错的。本组件只当 tab body，由 index.ts 在模式 ON 时 `openTab` 打开。
 *
 * 渲染：fetch cockpit HTTP 读原语（/api/config-tree 的 groups 二级分类 + /api/history），
 * 亮色科技感；任何语言合规项目只要暴露同款读原语即可（社区化目标）。
 */
import { useEffect, useRef } from 'react'
import { readBootstrap } from './bootstrap.ts'

const TAB_CSS = `
.re0-tab-native{height:100%;overflow:auto;background:#f6f8fa;color:#1f2328;font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.re0-tab-native .re0-native{display:flex;gap:8px;padding:8px;height:100%}
.re0-tab-native .re0-native-col{flex:1;overflow:auto;background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:10px}
.re0-tab-native h4{margin:0 0 8px;color:#0969da;font-size:11px;letter-spacing:.1em;text-transform:uppercase;border-bottom:1px solid #d8e6f5;padding-bottom:4px}
.re0-tab-native details{margin:2px 0 6px 6px;border-left:2px solid #d0d7de;padding-left:8px}
.re0-tab-native summary{cursor:pointer;color:#57606a;font-weight:600}
.re0-tab-native table{width:100%;border-collapse:collapse}
.re0-tab-native td{border-bottom:1px solid #eaeef2;padding:2px 4px;vertical-align:top}
.re0-tab-native td:first-child{color:#0550ae}
.re0-tab-native .re0-native-msg{padding:8px;color:#cf222e}
`

export function CockpitTabBody() {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // 注入一次性样式
    if (!document.querySelector('style[data-re0-tab-css]')) {
      const s = document.createElement('style')
      s.setAttribute('data-re0-tab-css', '1')
      s.textContent = TAB_CSS
      document.head.appendChild(s)
    }
    const root = ref.current
    if (!root) return
    const url = (readBootstrap()?.cockpitUrl || '').replace(/\/$/, '')
    const render = async (): Promise<void> => {
      if (!url) { root.innerHTML = '<div class="re0-native-msg">cockpitUrl 未注入</div>'; return }
      try {
        const t = await (await fetch(url + '/api/config-tree')).json()
        const h = await (await fetch(url + '/api/history?since_seq=0')).json()
        const groups = t.groups || {}
        const fams = Object.entries(groups).map(([fam, subs]: any) =>
          `<details open><summary>${fam}</summary>` +
          (subs || []).map((sg: any) =>
            `<details open><summary>${sg.prefix || '(top)'} (${(sg.rows || []).length})</summary><table>` +
            (sg.rows || []).map((r: any) =>
              `<tr><td title="${r.name}">${r.short || r.name}</td><td>${String(r.value)}</td></tr>`).join('') +
            '</table></details>').join('') +
          '</details>').join('')
        const evs = ((h.events || []).slice(-40)).map((e: any) =>
          `<tr><td>${e.seq}</td><td>${e.actor}</td><td>${e.kind}</td><td>${e.target}</td><td>${e.reason || ''}</td></tr>`).join('')
        root.innerHTML = `<div class="re0-native"><div class="re0-native-col"><h4>配置态</h4>${fams}</div>` +
          `<div class="re0-native-col"><h4>事件（最近 40）</h4><table>${evs}</table></div></div>`
      } catch {
        root.innerHTML = '<div class="re0-native-msg">原生驾驶舱读取失败</div>'
      }
    }
    render()
    const iv = setInterval(render, 3000)
    return () => clearInterval(iv)
  }, [])

  return <div ref={ref} className="re0-tab-native" />
}
