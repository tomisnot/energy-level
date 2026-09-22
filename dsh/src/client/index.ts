/**
 * re0-mecha dsh 插件 —— **CLIENT 入口**（浏览器侧 Cordis 插件）。
 *
 * 驾驶舱主界面 = dsh **右侧栏**：点 ◈驾驶舱 → `ctx.layout.openDetails()` 打开右栏 +
 * 把 cockpit iframe 挂进右栏列（`[data-rightbar-col]`）+ 自动关左栏；对话回中栏全高
 * （宽度随右栏打开自动收窄）。再点 → closeDetails + 复原左栏 + 移除 iframe。
 *
 * 贡献：① 头部切换按钮（`conversation.session.header.actions`）；② 注入 CSS
 * （`<style data-plugin-css>`，仅 `html[data-re0-cockpit="on"]` 生效）。
 * 跨 scope 的模式状态由模块级 mode-store 桥接；注册与副作用都经 `ctx.effect` 可逆。
 *
 * ⚠ client 半经 tsdown 打成 CJS + `__ModuleLoader__.load` 包装；改本目录源码后须
 * `npm run bundle` 并重启 dsh 才生效。
 */
import type { Context } from '@deepseek-ai/cordis'
import { CockpitPanel } from './CockpitPanel.tsx'
import { COCKPIT_MODE_CSS } from './cockpit-mode-css.ts'
import { getCockpitMode, subscribeCockpitMode } from './mode-store.ts'
import { readBootstrap } from './bootstrap.ts'

/** Cordis 插件名（与 host 半一致）。 */
export const name = 're0-mecha'

/** 原生驾驶舱 CSS（Phase 4-S1）：铺满右栏列 + 暗色 + 双栏（配置态/事件）。 */
const NATIVE_CSS = `
.re0-cockpit-native{position:absolute;inset:0;overflow:auto;background:#0d1117;color:#c9d1d9;font-size:12px;z-index:20}
.re0-cockpit-native .re0-native{display:flex;gap:8px;padding:8px;height:100%}
.re0-cockpit-native .re0-native-col{flex:1;overflow:auto;border:1px solid #30363d;border-radius:6px;padding:8px}
.re0-cockpit-native h4{margin:0 0 6px;color:#58a6ff}
.re0-cockpit-native table{width:100%;border-collapse:collapse}
.re0-cockpit-native td{border-bottom:1px solid #21262d;padding:2px 4px;vertical-align:top}
.re0-cockpit-native details summary{cursor:pointer;color:#8b949e;margin:4px 0}
.re0-cockpit-native .re0-native-msg{padding:8px;color:#f85149}
`

/** 需要 web client 的服务：slots（挂按钮）+ layout（开/关右栏、关左栏）。
 *  Cordis ctx 是代理：未在 inject 声明的服务一访问就抛
 *  "cannot get property X without inject"，故 layout 必须在此声明。 */
export const inject = ['slots', 'layout']

export async function apply(ctx: Context, config?: { cockpitNative?: boolean }): Promise<void> {
  await ctx.effect(() => {
    // 注入驾驶舱模式 CSS（data-plugin-css 标记；卸载即移除）
    const style = document.createElement('style')
    style.setAttribute('data-plugin-css', 're0-cockpit-mode')
    style.textContent = COCKPIT_MODE_CSS + NATIVE_CSS
    document.head.appendChild(style)

    // 会话头部：驾驶舱布局切换按钮
    const disposeBtn = ctx.slots.inject('conversation.session.header.actions', () =>
      ctx.slots.register(
        { name: 'conversation.session.header.actions', id: 're0-cockpit', order: 200 },
        CockpitPanel,
      ))

    // ---- 模式副作用：右栏=驾驶舱 + 左栏自动关 + 驾驶舱挂右栏列 ----
    // 双模（Phase 4-S1）：config.cockpitNative!==false → **原生渲染**（fetch cockpit
    // HTTP 读原语 /api/config-tree + /api/history，不 iframe，任何语言合规项目一键接）；
    // ===false → 回退 iframe（cockpitUrl）。
    const useNative = config?.cockpitNative !== false
    let frame: HTMLIFrameElement | null = null
    let nativeRoot: HTMLDivElement | null = null
    let poll: ReturnType<typeof setInterval> | null = null
    let weClosedSidebar = false

    const renderNative = async (root: HTMLDivElement): Promise<void> => {
      const url = (readBootstrap()?.cockpitUrl || '').replace(/\/$/, '')
      if (!url) { root.innerHTML = '<div class="re0-native-msg">cockpitUrl 未注入</div>'; return }
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
        root.innerHTML = '<div class="re0-native-msg">原生驾驶舱读取失败（可 config.cockpitNative=false 回退 iframe）</div>'
      }
    }
    const mountNative = (): void => {
      if (nativeRoot) return
      // openRightbar 触发 React 重渲染，[data-rightbar-col] **异步**才出现；同步查会落空
      // → 轮询等列出现再挂（~4s 放弃；模式已 OFF 也放弃）。
      let tries = 0
      const iv = setInterval(() => {
        tries++
        const host = document.querySelector('[data-rightbar-col]')
        if (host) {
          clearInterval(iv)
          nativeRoot = document.createElement('div')
          nativeRoot.className = 're0-cockpit-native'
          host.appendChild(nativeRoot)
          renderNative(nativeRoot)
          poll = setInterval(() => { if (nativeRoot) renderNative(nativeRoot) }, 3000)
        } else if (tries > 40 || !getCockpitMode()) {
          clearInterval(iv)
        }
      }, 100)
    }
    const unmountNative = (): void => {
      if (poll) { clearInterval(poll); poll = null }
      if (nativeRoot) { nativeRoot.remove(); nativeRoot = null }
    }
    const mountFrame = (): void => {
      if (frame) return
      let tries = 0
      const iv = setInterval(() => {
        tries++
        const host = document.querySelector('[data-rightbar-col]')
        if (host) {
          clearInterval(iv)
          frame = document.createElement('iframe')
          frame.className = 're0-cockpit-frame'
          const url = readBootstrap()?.cockpitUrl
          if (url) frame.src = url
          frame.title = 'Re0 机甲驾驶舱（只读）'
          host.appendChild(frame)
        } else if (tries > 40 || !getCockpitMode()) {
          clearInterval(iv)
        }
      }, 100)
    }
    const unmountFrame = (): void => {
      if (frame) { frame.remove(); frame = null }
    }

    const unsub = subscribeCockpitMode(() => {
      if (getCockpitMode()) {
        // 运行版 dsh(0.1.5-rc.2) 的 ctx.layout 是 openRightbar/closeRightbar（非 openDetails）；
        // 全部可选链：API 缺失时降级不崩整站。openRightbar(true,false)=预留右栏 grid track。
        ctx.layout?.openRightbar?.(true, false)
        // 左栏若开着则关掉（记住是我们关的，OFF 时复原）
        const fr = document.querySelector('[data-re0-frame]')
        if (fr && !fr.hasAttribute('data-sidebar-collapsed')) {
          ctx.layout?.toggleSidebar?.()
          weClosedSidebar = true
        }
        if (useNative) mountNative(); else mountFrame()
      } else {
        ctx.layout?.closeRightbar?.()
        if (weClosedSidebar) { ctx.layout?.toggleSidebar?.(); weClosedSidebar = false }
        unmountNative()
        unmountFrame()
      }
    })

    return () => {
      unsub()
      unmountNative()
      unmountFrame()
      try { disposeBtn?.() } catch { /* ignore */ }
      style.remove()
      document.documentElement.removeAttribute('data-re0-cockpit')
    }
  }, 're0-mecha: cockpit mode')
}
