/**
 * 右栏驾驶舱 occupant（Phase 4-S1 修正）：注册进 `rightbar.session` slot。
 *
 * 关键：dsh 的 `RightbarColumn`（`[data-rightbar-col]`）**只在右栏有 occupant 时才渲染**；
 * 先前只 DOM-append 而不注册 occupant → 列根本不出现 → append 落空、右栏开但空白。
 * 故驾驶舱内容改由本 occupant 组件**自己渲染**（native：fetch cockpit HTTP 读原语）。
 *
 * 双模：`globalThis.__RE0_COCKPIT_NATIVE__ === false` → iframe（cockpitUrl）兜底；否则 native。
 */
import { useEffect, useRef } from 'react'
import { readBootstrap } from './bootstrap.ts'

export function CockpitRightbar() {
  const ref = useRef<HTMLDivElement>(null)
  const native = (globalThis as any).__RE0_COCKPIT_NATIVE__ !== false

  useEffect(() => {
    const root = ref.current
    if (!root || !native) return
    const url = (readBootstrap()?.cockpitUrl || '').replace(/\/$/, '')
    const render = async (): Promise<void> => {
      if (!url) {
        root.innerHTML = '<div class="re0-native-msg">cockpitUrl 未注入</div>'
        return
      }
      try {
        const t = await (await fetch(url + '/api/config-tree')).json()
        const h = await (await fetch(url + '/api/history?since_seq=0')).json()
        const fams = Object.entries(t.families || {}).map(([fam, rows]: any) =>
          `<details open><summary>${fam} (${rows.length})</summary><table>${
            (rows as any[]).map((r) => `<tr><td>${r.name}</td><td>${String(r.value)}</td></tr>`).join('')
          }</table></details>`).join('')
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
  }, [native])

  const url = readBootstrap()?.cockpitUrl
  if (!native) {
    return <iframe className="re0-cockpit-frame" src={url} title="Re0 机甲驾驶舱（只读）"
      style={{ width: '100%', height: '100%', border: 'none' }} />
  }
  return <div ref={ref} className="re0-cockpit-native" />
}
