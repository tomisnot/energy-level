/**
 * re0-mecha dsh 插件 —— **CLIENT 入口**（浏览器侧 Cordis 插件）。
 *
 * 两模式收拢（2026-09-24）后重建的 dsh 界面，三件事（第三件是**能力开关**）：
 *  1. ◈监控 按钮（会话头部）→ 开右栏 + 打开 `re0-cockpit` tab（旧覆盖层重布局
 *     已弃，右栏 tab 由 dsh 自管开合，按钮只负责"打开"，最轻）。
 *  2. 右栏 tab body（`MonitorTabBody`）→ 轮询权威**只读监控端点**的
 *     `/history`（写事件，含 `mode`）+ `/config`（配置态树），显示"当前写权模式 +
 *     AI/人最近改了什么"。
 *  3. ⭐ **切换至 GUI 按钮**（会话头部）→ 经 `/re0-gui` command 让 host 写 .mode-request，
 *     看门人接手 terminate 权威+dsh、起 GUI。**它由能力开关
 *     `EL_FEATURES.enableSwitchToGui` 控制**（见 `el-params.ts`）：关着时按钮与 command
 *     **都不存在**（两半读**同一个** `headerPlan`）——没有看门人的部署不该出现
 *     "按了没反应"的按钮。
 *
 * **地址不再硬编码**（2026-09-25 第 2 段）：监控端点是**动态端口**，面板经共享资产的
 * `fetchMonitorBase` 去 host 半的**同源只读路由**取 `{base}`（host 读权威写的端口文件）。
 * 每一轮轮询都现取 ⇒ 权威重启换端口后下一拍即落在新端口，不需要重启 dsh、
 * 更不会"面板空白但没人报错"（空数据由 `empty` 档显式表达，不伪装成成功）。
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
import { MonitorTabBody } from '../dsh-panel/MonitorTabBody.tsx'
import { CockpitButton, SwitchToGuiButton, type CockpitInjected, type SwitchInjected } from './buttons.tsx'
import { PANEL_CONFIG } from '../dsh-panel/panel-config.ts'
import { EL_FEATURES, EL_TAB, headerPlan } from '../el-params.ts'

/** Cordis 插件名（与 host 半一致）。 */
export const name = 're0-mecha'

/** 会话头部动作的槽位名（本半两处注册同一个槽）。 */
const HEADER_SLOT = 'conversation.session.header.actions'

/** 本半的动作计划——唯一权威是 `el-params.ts` 的 `headerPlan`（host 半读同一个）。 */
const PLAN = headerPlan(EL_FEATURES.enableSwitchToGui)

/** slots（按钮+tab body）+ layout（开右栏）+ 右栏 tab 服务；交权按钮另需 remote.commands。 */
export const inject = [
  'slots', 'layout', 'sidebarRight', 'sidebarRightTabs',
  ...(PLAN.switchToGui ? ['remote', 'remote.commands'] : []),
]

export async function apply(ctx: Context): Promise<void> {
  await ctx.effect(() => {
    // 会话头部：◈监控 按钮（开右栏 + 打开我们的 tab）。sessionId 是组件 standardProp，
    // inject 只暴露"无状态动作"，组件用自己的 props.sessionId 调需要它的动作。
    const disposeCockpit = ctx.slots.inject(HEADER_SLOT, () =>
      ctx.slots.register(
        { name: HEADER_SLOT, id: PLAN.cockpit.id, order: PLAN.cockpit.order,
          label: PLAN.cockpit.label,
          inject: (): CockpitInjected => ({
            openCockpit: () => {
              ctx.layout?.openRightbar?.(true, false)
              try { (ctx as any).sidebarRight?.openTab?.(EL_TAB.id) } catch { /* 降级 */ }
            },
          }) },
        CockpitButton,
      ))

    // 会话头部：切换至 GUI 按钮（发 /re0-gui command，host 写 .mode-request）。
    // switchToGui 收 sessionId（组件传自己的 props.sessionId）。
    // ⚠ **能力开关**：关着时这一段整体不执行（`disposeSwitch` 保持 null）⇒ 不留按钮；
    // command 那侧由 host 半按**同一个计划**同步不注册 ⇒ 不会出现"按钮在、command 不在"。
    const sw = PLAN.switchToGui
    const disposeSwitch = sw
      ? ctx.slots.inject(HEADER_SLOT, () =>
          ctx.slots.register(
            { name: HEADER_SLOT, id: sw.id, order: sw.order, label: sw.label,
              inject: (): SwitchInjected => ({
                switchToGui: async (sessionId: SessionId) => {
                  const line = `/${sw.command}`
                  const r = await ctx.remote!.commands!.execute(sessionId, line, [])
                  if (!r.ok) return r.error.message
                  if (r.value === undefined) return `unknown command: ${line}`
                  return null
                },
              }) },
            SwitchToGuiButton,
          ))
      : null

    // 右栏 tab 类型 + body（body 由 dsh 在该 tab 打开时渲染）。
    // 标题取**参数块**的 `TITLE`（此前是就地手抄的字面量 ⇒ 参数块里那一项没有消费者）。
    const relType = (ctx as any).sidebarRightTabs?.register?.({
      id: EL_TAB.id, kind: EL_TAB.id, title: () => PANEL_CONFIG.TITLE,
    }) || null
    const disposeBody = ctx.slots.inject('sidebar.right.pane.tab', () =>
      ctx.slots.register(
        // 面板**自己取数**（原生形态）：取址/取数/归约/渲染全在共享资产里
        // （`dsh-panel/{panel-data,panel-view,MonitorTabBody}`）⇒ **没有注入面**
        // （原先那个 `resolveMonitorBase` 缝已被兜住；留着就是"声明了却不被读"的死缝）。
        // 本仓只提供**项目自己的东西**：参数块（路由/端口文件/标题）+ 下面的 tab 注册。
        { name: 'sidebar.right.pane.tab', key: EL_TAB.id } as any,
        MonitorTabBody as any,
      ))

    return () => {
      if (disposeBody) { try { disposeBody() } catch { /* ignore */ } }
      if (typeof relType === 'function') { try { relType() } catch { /* ignore */ } }
      if (disposeSwitch) { try { disposeSwitch() } catch { /* ignore */ } }
      try { disposeCockpit?.() } catch { /* ignore */ }
    }
  }, 're0-mecha: client monitor UI')
}
