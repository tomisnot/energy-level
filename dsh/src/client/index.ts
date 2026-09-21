/**
 * re0-mecha dsh 插件 —— **CLIENT 入口**（浏览器侧 Cordis 插件）。
 *
 * 贡献两处 UI + 注入一段 CSS：
 * 1. `conversation.session.header.actions`（single/list, session）：◈驾驶舱 **切换按钮**（CockpitPanel）。
 * 2. `shell.overlay`（list, root）：驾驶舱 **iframe 覆盖层**（CockpitOverlay），模式 ON 时填满。
 * 3. `<style data-plugin-css>`：驾驶舱模式 CSS——隐藏左侧栏、把中栏（AI 对话）fixed 到右上角
 *    约 1/4、驾驶舱铺满覆盖层。全部限定在 `html[data-re0-cockpit="on"]` 下，默认原版布局零影响。
 *
 * 跨 scope 的模式状态由模块级 mode-store 桥接（按钮 session、覆盖层 root）。注册与 DOM 副作用
 * 都经 `ctx.effect` 可逆：插件卸载即移除按钮/覆盖层/<style>，并复位模式（清 html 属性、断观察器）。
 *
 * ⚠ client 半经 tsdown 打成 CJS + `__ModuleLoader__.load` 包装（见 tsdown.config.ts）；改本目录
 * 源码后须 `npm run bundle` 重新生成 lib/client.js 才在 dsh 里生效。ESM 顶层 import/export 会毒化
 * dsh 的合并包（整站崩），故 react / @deepseek-ai/cordis 等按 platform external 处理。
 */
import type { Context } from '@deepseek-ai/cordis'
import { CockpitPanel } from './CockpitPanel.tsx'
import { CockpitOverlay } from './CockpitOverlay.tsx'
import { COCKPIT_MODE_CSS } from './cockpit-mode-css.ts'
import { setCockpitMode } from './mode-store.ts'

/** Cordis 插件名（与 host 半一致）。 */
export const name = 're0-mecha'

/** 需要 web client 的 slots 服务。 */
export const inject = ['slots']

export async function apply(ctx: Context): Promise<void> {
  await ctx.effect(() => {
    // 注入驾驶舱模式 CSS（data-plugin-css 标记；卸载即移除）
    const style = document.createElement('style')
    style.setAttribute('data-plugin-css', 're0-cockpit-mode')
    style.textContent = COCKPIT_MODE_CSS
    document.head.appendChild(style)

    // 会话头部：驾驶舱布局切换按钮
    const disposeBtn = ctx.slots.inject('conversation.session.header.actions', () =>
      ctx.slots.register(
        { name: 'conversation.session.header.actions', id: 're0-cockpit', order: 200 },
        CockpitPanel,
      ))

    // 壳级覆盖层：驾驶舱 iframe（list 槽，ON 时渲染，OFF 时组件返回 null）
    const disposeOverlay = ctx.slots.inject('shell.overlay', () =>
      ctx.slots.register(
        { name: 'shell.overlay', id: 're0-cockpit', order: 100 },
        CockpitOverlay,
      ))

    return () => {
      try { disposeBtn?.() } catch { /* ignore */ }
      try { disposeOverlay?.() } catch { /* ignore */ }
      setCockpitMode(false)                 // 清 html 属性 + 断开列标记观察器
      style.remove()
    }
  }, 're0-mecha: cockpit mode')
}
