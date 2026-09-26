/**
 * re0-mecha dsh 插件 —— **CLIENT 入口**（浏览器侧 Cordis 插件）。
 *
 * 两模式收拢（2026-09-24）后重建的 dsh 界面，三件事：
 *  1. ◈监控 按钮（会话头部）→ 开右栏 + 打开 `re0-cockpit` tab（旧覆盖层重布局
 *     已弃，右栏 tab 由 dsh 自管开合，按钮只负责"打开"，最轻）。
 *  2. 右栏 tab body（`MonitorTabBody`）→ 轮询权威**只读监控端点**的
 *     `/history`（写事件，含 `mode`）+ `/config`（配置态树），显示"当前写权模式 +
 *     AI/人最近改了什么"。
 *  3. 切换至 GUI 按钮（会话头部）→ 经 `/re0-gui` command 让 host 写 .mode-request，
 *     看门人接手 terminate 权威+dsh、起 GUI。client 不能直接调 MCP 工具/写文件，
 *     只能发 command（dsh 事实：ui-plan 的 exitPlanMode 同款，经 sessionId 注入）。
 *
 * **地址不再硬编码**（2026-09-25 第 2 段）：监控端点是**动态端口**，面板经注入的
 * `resolveMonitorBase()` 去 host 半的**同源只读路由**取 `{base}`
 * （host 读权威写的端口文件）。每一轮轮询都现解析 ⇒ 权威重启换端口后下一拍即落在
 * 新端口，不需要重启 dsh、更不会"面板空白但没人报错"。
 *
 * **路由与端口文件名不再是本仓常量**（2026-09-25 共享资产迁移）：两半都从参数块
 * `./../dsh-panel/panel-config.ts` 的 `PANEL_CONFIG` 取——两半同源同包 ⇒ "单一来源"
 * 在包内自动满足（参数块不 import `node:*`，故 client 半也能安全 import）。
 *
 * ⚠ client 半经 tsdown 打成 CJS + `__ModuleLoader__.load` 包装；改本目录源码后须
 * `npm run bundle` 并重启 dsh 才生效。
 */
import type { Context } from '@deepseek-ai/cordis'
import type { SessionId } from '@deepseek-ai/dsh-session/types'
import { MonitorTabBody } from './MonitorTabBody.tsx'
import { CockpitButton, SwitchToGuiButton, type CockpitInjected, type SwitchInjected } from './buttons.tsx'
import { PANEL_CONFIG } from '../dsh-panel/panel-config.ts'

/** Cordis 插件名（与 host 半一致）。 */
export const name = 're0-mecha'

/** slots（按钮+tab body）+ layout（开右栏）+ 右栏 tab 服务 + remote.commands（发 /re0-gui）。 */
export const inject = ['slots', 'layout', 'sidebarRight', 'sidebarRightTabs', 'remote', 'remote.commands']

/** host 半注册的同源只读路由（监控地址查询）。**来源 = 参数块**（两半同一常量）。 */
export const ADDRESS_ROUTE = PANEL_CONFIG.ROUTE_PATH

/** tab body 的注入面：解析监控端点基址（由 host 半的同源路由提供）。 */
export interface MonitorInjected {
  resolveMonitorBase(): Promise<string>
}

/**
 * 取监控端点基址：同源 `fetch` → `{base}`。
 *
 * 任何一步不对都**抛错**（面板按"连不上权威"显示）——**不回落到任何默认端口**：
 * 回落会把"权威没起来"伪装成"连上了但空面板"。`doFetch` 可注入，判据在 Node 里
 * 喂真形状与各种失败形状。
 */
export async function fetchMonitorBase(
  doFetch: typeof fetch = fetch,
): Promise<string> {
  const res = await doFetch(ADDRESS_ROUTE, { cache: 'no-store' })
  if (!res.ok) throw new Error(`地址路由返回 ${res.status}`)
  const data: any = await res.json()
  const base = data && typeof data.base === 'string' ? data.base : ''
  if (!base) throw new Error('地址路由的返回里没有 base（形状不对？）')
  return base.trim().replace(/\/+$/, '')
}

export async function apply(ctx: Context): Promise<void> {
  await ctx.effect(() => {
    // 会话头部：◈监控 按钮（开右栏 + 打开我们的 tab）。sessionId 是组件 standardProp，
    // inject 只暴露"无状态动作"，组件用自己的 props.sessionId 调需要它的动作。
    const disposeCockpit = ctx.slots.inject('conversation.session.header.actions', () =>
      ctx.slots.register(
        { name: 'conversation.session.header.actions', id: 're0-cockpit', order: 200, label: '◈ 监控',
          inject: (): CockpitInjected => ({
            openCockpit: () => {
              ctx.layout?.openRightbar?.(true, false)
              try { (ctx as any).sidebarRight?.openTab?.('re0-cockpit') } catch { /* 降级 */ }
            },
          }) },
        CockpitButton,
      ))

    // 会话头部：切换至 GUI 按钮（发 /re0-gui command，host 写 .mode-request）。
    // switchToGui 收 sessionId（组件传自己的 props.sessionId）。
    const disposeSwitch = ctx.slots.inject('conversation.session.header.actions', () =>
      ctx.slots.register(
        { name: 'conversation.session.header.actions', id: 're0-to-gui', order: 201, label: '→ GUI',
          inject: (): SwitchInjected => ({
            switchToGui: async (sessionId: SessionId) => {
              const r = await ctx.remote!.commands!.execute(sessionId, '/re0-gui', [])
              if (!r.ok) return r.error.message
              if (r.value === undefined) return 'unknown command: /re0-gui'
              return null
            },
          }) },
        SwitchToGuiButton,
      ))

    // 右栏 tab 类型 + body（body 由 dsh 在该 tab 打开时渲染）
    const relType = (ctx as any).sidebarRightTabs?.register?.({
      id: 're0-cockpit', kind: 're0-cockpit', title: () => '机甲监控',
    }) || null
    const disposeBody = ctx.slots.inject('sidebar.right.pane.tab', () =>
      ctx.slots.register(
        { name: 'sidebar.right.pane.tab', key: 're0-cockpit',
          // 注入面：面板拿它取监控地址（host 半的同源只读路由）。
          inject: (): MonitorInjected => ({ resolveMonitorBase: fetchMonitorBase }) } as any,
        MonitorTabBody as any,
      ))

    return () => {
      if (disposeBody) { try { disposeBody() } catch { /* ignore */ } }
      if (typeof relType === 'function') { try { relType() } catch { /* ignore */ } }
      try { disposeSwitch?.() } catch { /* ignore */ }
      try { disposeCockpit?.() } catch { /* ignore */ }
    }
  }, 're0-mecha: client monitor UI')
}
