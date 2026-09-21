/**
 * re0-mecha dsh 插件 —— **CLIENT 入口**（浏览器侧 Cordis 插件）。
 *
 * 把一个"驾驶舱"动作挂到会话头部槽（`conversation.session.header.actions`），点开 iframe
 * 嵌 cockpit-server（配置态树 / 事件流 / 告警）。姿势照 `client-ui-agent-team`（slots.inject
 * → slots.register）+ `slots.zh.md` 的逐字示例。注册经 `ctx.effect` 可逆（卸载即移除）。
 *
 * ⚠ out-of-tree 的 client 半如何被 dsh web client loader 挂载（package.json 的 `dsh.client`
 * + `/client` 导出）需活体确认——in-tree 范例（inspector/agent-team）走 workspace 装配。
 */
import type { Context } from '@deepseek-ai/cordis'
import { CockpitPanel } from './CockpitPanel.tsx'

/** Cordis 插件名（与 host 半一致）。 */
export const name = 're0-mecha'

/** 需要 web client 的 slots 服务。 */
export const inject = ['slots']

export async function apply(ctx: Context): Promise<void> {
  await ctx.effect(() => {
    const dispose = ctx.slots.inject('conversation.session.header.actions', () =>
      ctx.slots.register(
        { name: 'conversation.session.header.actions', id: 're0-cockpit', order: 200 },
        CockpitPanel,
      ))
    return () => {
      try { dispose?.() } catch { /* ignore */ }
    }
  }, 're0-mecha: cockpit panel')
}
