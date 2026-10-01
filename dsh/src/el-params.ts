/**
 * EL 的**参数家**（本项目专有；**不进共享资产指纹表**）。
 *
 * ## 为什么单独立一个文件
 *
 * 共享资产 `dsh-panel/` 把"可变量"收进它自己的参数块（`dsh-panel/panel-config.ts`），
 * 而**资产覆盖不到的项目值**——MCP 桥的日志前缀 / 断联指引 / 客户端标识 / 端点默认，
 * 以及「交权按钮」这个**能力开关**——也需要**一个家**。散在 host/client 两半的各调用点上
 * 就是两处真值，正是本工程反复付代价的那类漂移（R1 / D1）。
 *
 * ## 硬约束：本文件必须**浏览器安全**
 *
 * client 半会 import 它（交权按钮的开关）⇒ **零 import、零 `node:*`**。
 * 需要读文件或碰 SDK 的 EL 参数填法在 `host/el-hub.ts`（node-only）。
 *
 * ## 与资产的关系
 *
 * 资产件一律**逐字复制**（含 `mcp-bridge.ts` / `register-tools.ts` / `config.ts` /
 * `mcp-session-http.ts`）；EL 的值**只在这里与 `host/el-hub.ts` 出现**，经参数传进去
 * ⇒ "逐字复制"才成立（参数表见 `dsh-panel/README.md`）。
 */

/** EL 填给共享资产 opt-in 件的值（**唯一来源**）。 */
export const EL_BRIDGE = {
  /** `BridgeOptions.logLabel`：日志前缀 `[re0-mecha] …`（与 EL 旧的日志前缀逐字一致）。 */
  logLabel: 're0-mecha',
  /**
   * `BridgeOptions.offlineHint`：断联时那句**可操作指引**（项目知识：怎么把软件重新起来）。
   * 原自建版把这段写死在桥的错误文案里；资产把它参数化了 ⇒ 必须由 EL 显式传，
   * 否则退回中性句（"请确认软件已启动…"），我们就丢了"要人经 launcher 起 AI 模式"这条指引。
   */
  offlineHint: '请确认软件在 AI 模式（人经 launcher.py 启动）；工具仍在列表但暂不可用。',
  /** `HttpSessionOptions.clientInfo`：initialize 上报的客户端标识。 */
  clientInfo: { name: 're0-mecha-dsh', version: '0.1.0' },
  /** `HubUrlConfig.defaultUrl`：端口文件读不到时的默认端点（原自建版的 `DEFAULT_URL`）。 */
  defaultUrl: 'http://127.0.0.1:8765/mcp',
  /** `HubUrlConfig.mcpPortFile`：权威写的 MCP 端口文件名（原自建版的默认值）。 */
  mcpPortFile: '.mcp-port',
} as const

/**
 * EL 的能力开关（**不是取值，是能力**）。
 *
 * `enableSwitchToGui`：会话头部的「→ GUI」按钮 + host 侧的 `/re0-gui` command。
 * ⚠ **它的前提是"有人在轮询 `.mode-request`"**——EL 的部署恒经看门人（`launcher.py`，
 * 见 `docs/交接-给下一代.md` §2），所以 EL 开；**没有看门人的部署必须关**，
 * 否则按钮按了没人接（"按了没反应"是本工程明确要避免的形态）。
 *
 * ⚠ **为什么是一个源级常量、而不是 dsh 插件的运行期 config**：client 半**未必**拿得到
 * entry 的 `config`（本 dsh 版本**未核**）；若 host 半按 config 关、client 半按常量开，
 * 就会出现"按钮在、command 不在"——正是我们要禁的死按钮。⇒ **两半读同一个常量**，
 * 一处关就两处都关（开关的唯一权威在本文件）。
 */
export const EL_FEATURES = {
  enableSwitchToGui: true,
} as const

/** 会话头部两个动作的**标签**（唯一来源：`buttons.tsx` 与槽位注册都用它，不许手抄两遍）。 */
export const EL_HEADER_LABEL = {
  cockpit: '◈ 监控',
  switchToGui: '→ GUI',
} as const

/** 右栏监控 tab 的 id / kind（dsh 的 `sidebarRightTabs` 与本仓的 slot key 必须同串）。 */
export const EL_TAB = {
  id: 're0-cockpit',
} as const

/** 一个头部动作的注册计划（槽位 id/order/label）。 */
export interface HeaderAction {
  readonly id: string
  readonly order: number
  readonly label: string
}

/** 交权动作：**带 command**（host 半据此决定注册哪个 slash command）。 */
export interface SwitchAction extends HeaderAction {
  readonly command: string
}

/** 头部动作计划：两半消费**同一个计划** ⇒ 不可能一处开一处关。 */
export interface HeaderPlan {
  /** 监控按钮（开右栏 + 打开面板 tab）——**永远有**。 */
  readonly cockpit: HeaderAction
  /** 交权按钮 + 它的 command——**开关关着时是 `null`**。 */
  readonly switchToGui: SwitchAction | null
}

/**
 * 按能力开关算出"会话头部要注册哪些动作"（纯函数，可单测；两半共用）。
 *
 * ⚠ 关着时返回 `switchToGui: null` = **不注册按钮、也不注册 command**（R17：不留死缝，
 * 也不留按了没反应的按钮）。
 */
export function headerPlan(enableSwitchToGui: boolean): HeaderPlan {
  return {
    cockpit: { id: EL_TAB.id, order: 200, label: EL_HEADER_LABEL.cockpit },
    switchToGui: enableSwitchToGui
      ? { id: 're0-to-gui', order: 201, label: EL_HEADER_LABEL.switchToGui,
          command: 're0-gui' }
      : null,
  }
}
