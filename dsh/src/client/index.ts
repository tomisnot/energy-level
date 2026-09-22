/**
 * re0-mecha dsh 插件 —— **CLIENT 入口**（浏览器侧 Cordis 插件）。
 *
 * 驾驶舱主界面 = dsh **右侧栏**：点 ◈驾驶舱 → `ctx.layout.openRightbar(true,false)` 开右栏
 * + 把 `CockpitRightbar` 注册进 `rightbar.session` slot（**右栏列只在有 occupant 时渲染**，
 * 故必须注册 occupant 而非 DOM-append 到可能不存在的列）+ 自动关左栏。再点 → 注销 occupant
 * + closeRightbar + 复原左栏。
 *
 * 驾驶舱内容双模（Phase 4-S1）：默认 **native**（occupant 内 fetch cockpit HTTP 读原语
 * /api/config-tree + /api/history 渲染，任何语言合规项目一键接）；`config.cockpitNative===false`
 * → occupant 渲染 iframe（cockpitUrl）兜底。
 *
 * 贡献：① 头部切换按钮（`conversation.session.header.actions`）；② 右栏 occupant；
 * ③ 注入 CSS（`<style data-plugin-css>`）。跨 scope 模式状态由模块级 mode-store 桥接；
 * 注册与副作用都经 `ctx.effect` 可逆。
 *
 * ⚠ client 半经 tsdown 打成 CJS + `__ModuleLoader__.load` 包装；改本目录源码后须
 * `npm run bundle` 并重启 dsh 才生效。
 */
import type { Context } from '@deepseek-ai/cordis'
import { CockpitPanel } from './CockpitPanel.tsx'
import { CockpitRightbar } from './CockpitRightbar.tsx'
import { COCKPIT_MODE_CSS } from './cockpit-mode-css.ts'
import { getCockpitMode, subscribeCockpitMode } from './mode-store.ts'

/** Cordis 插件名（与 host 半一致）。 */
export const name = 're0-mecha'

/** 原生驾驶舱 CSS：**亮色科技感**（与 dsh 亮色主界面搭）：浅灰底+白卡片+等宽字+
 * 大写蓝色栏目头+细边框阴影。occupant 容器内铺满 + 双栏（配置态/事件）。 */
const NATIVE_CSS = `
.re0-cockpit-native{height:100%;overflow:auto;background:#f6f8fa;color:#1f2328;font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.re0-cockpit-native .re0-native{display:flex;gap:8px;padding:8px;height:100%}
.re0-cockpit-native .re0-native-col{flex:1;overflow:auto;background:#ffffff;border:1px solid #d0d7de;border-radius:8px;padding:10px;box-shadow:0 1px 2px rgba(31,35,40,.05)}
.re0-cockpit-native h4{margin:0 0 8px;color:#0969da;font-size:11px;letter-spacing:.1em;text-transform:uppercase;border-bottom:1px solid #d8e6f5;padding-bottom:4px}
.re0-cockpit-native table{width:100%;border-collapse:collapse}
.re0-cockpit-native td{border-bottom:1px solid #eaeef2;padding:2px 4px;vertical-align:top}
.re0-cockpit-native td:first-child{color:#0550ae}
.re0-cockpit-native details summary{cursor:pointer;color:#57606a;margin:4px 0;font-weight:600}
.re0-cockpit-native .re0-native-msg{padding:8px;color:#cf222e}
`

/** 需要 web client 的服务：slots（挂按钮+右栏 occupant）+ layout（开/关右栏、关左栏）。
 *  Cordis ctx 是代理：未在 inject 声明的服务一访问就抛
 *  "cannot get property X without inject"，故 layout 必须在此声明。 */
export const inject = ['slots', 'layout']

export async function apply(ctx: Context, config?: { cockpitNative?: boolean }): Promise<void> {
  // 双模开关传给 occupant 组件（slots.register 的 props 由 dsh 给，故走 globalThis）
  ;(globalThis as any).__RE0_COCKPIT_NATIVE__ = config?.cockpitNative !== false

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

    // ---- 驾驶舱 = main 全局面板（keyed slot）----
    // 右栏是 tab 体制（列宽由 sidebar-right store 驱动）：shadow 掉 rightbar.session 后
    // 无激活 tab → 列宽塌 0、连空间都不出现。改用**可靠的 main keyed 面板**：注册一次，
    // 模式 ON 用 selectPanel('re0-cockpit') 切到驾驶舱（主区整屏），OFF selectPanel(null) 回对话。
    const disposePanel = ctx.slots.inject('main', () =>
      ctx.slots.register({ name: 'main', key: 're0-cockpit' } as any, CockpitRightbar))

    const unsub = subscribeCockpitMode(() => {
      if (getCockpitMode()) ctx.layout?.selectPanel?.('re0-cockpit')
      else ctx.layout?.selectPanel?.(null)
    })

    return () => {
      unsub()
      try { ctx.layout?.selectPanel?.(null) } catch { /* ignore */ }
      try { disposePanel?.() } catch { /* ignore */ }
      try { disposeBtn?.() } catch { /* ignore */ }
      style.remove()
      document.documentElement.removeAttribute('data-re0-cockpit')
    }
  }, 're0-mecha: cockpit mode')
}
