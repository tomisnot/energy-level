/**
 * re0-mecha dsh 插件 —— **CLIENT 入口**（浏览器侧 Cordis 插件）。
 *
 * 驾驶舱 = 右栏一个 **tab**（dockkit tab 体制，dsh 源码验证的正确路径）：注册 tab 类型
 * `re0-cockpit` + body（`CockpitTabBody`，native 渲染 cockpit HTTP 读原语）。点 ◈驾驶舱 →
 * openRightbar + 关左栏 + openTab('re0-cockpit')；再点 → closeRightbar + 复原左栏。
 *
 * **无 iframe、单一实现**；样式由 CockpitTabBody 自注入。
 *
 * ⚠ client 半经 tsdown 打成 CJS + `__ModuleLoader__.load` 包装；改本目录源码后须
 * `npm run bundle` 并重启 dsh 才生效。
 */
import type { Context } from '@deepseek-ai/cordis'
import { CockpitPanel } from './CockpitPanel.tsx'
import { CockpitTabBody } from './CockpitTabBody.tsx'
import { getCockpitMode, subscribeCockpitMode } from './mode-store.ts'

/** Cordis 插件名（与 host 半一致）。 */
export const name = 're0-mecha'

/** slots（按钮+tab body）+ layout（开/关右栏、关左栏）+ 右栏 tab 服务。 */
export const inject = ['slots', 'layout', 'sidebarRight', 'sidebarRightTabs']

export async function apply(ctx: Context): Promise<void> {
  await ctx.effect(() => {
    // 会话头部：驾驶舱切换按钮
    const disposeBtn = ctx.slots.inject('conversation.session.header.actions', () =>
      ctx.slots.register(
        { name: 'conversation.session.header.actions', id: 're0-cockpit', order: 200 },
        CockpitPanel,
      ))

    // 右栏 tab 类型 + body（注册一次；body 由 dsh 在该 tab 打开时渲染）
    const relType = (ctx as any).sidebarRightTabs?.register?.({
      id: 're0-cockpit', kind: 're0-cockpit', title: () => '机甲驾驶舱',
    }) || null
    const disposeBody = ctx.slots.inject('sidebar.right.pane.tab', () =>
      ctx.slots.register(
        { name: 'sidebar.right.pane.tab', key: 're0-cockpit' } as any,
        CockpitTabBody,
      ))

    // 模式副作用：ON → 开右栏+关左栏+开我们的 tab；OFF → 关右栏+复原左栏
    let weClosedSidebar = false
    const unsub = subscribeCockpitMode(() => {
      if (getCockpitMode()) {
        ctx.layout?.openRightbar?.(true, false)
        const fr = document.querySelector('[data-re0-frame]')
        if (fr && !fr.hasAttribute('data-sidebar-collapsed')) {
          ctx.layout?.toggleSidebar?.()
          weClosedSidebar = true
        }
        try { (ctx as any).sidebarRight?.openTab?.('re0-cockpit') } catch { /* 降级 */ }
      } else {
        ctx.layout?.closeRightbar?.()
        if (weClosedSidebar) { ctx.layout?.toggleSidebar?.(); weClosedSidebar = false }
      }
    })

    return () => {
      unsub()
      if (disposeBody) { try { disposeBody() } catch { /* ignore */ } }
      if (typeof relType === 'function') { try { relType() } catch { /* ignore */ } }
      try { disposeBtn?.() } catch { /* ignore */ }
    }
  }, 're0-mecha: cockpit mode')
}
