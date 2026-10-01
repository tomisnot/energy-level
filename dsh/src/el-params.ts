/**
 * EL 的**参数家**（本项目专有；**不进共享资产指纹表**）。
 *
 * ## 为什么单独立一个文件
 *
 * **资产覆盖不到的项目值**需要一个家：MCP 桥的日志前缀 / 断联指引 / 客户端标识 / 端点默认、
 * 面板的三项（路由 / 端口文件名 / 标题），以及「交权按钮」这个**能力开关**。
 * 散在 host/client 两半的各调用点上就是两处真值，正是本工程反复付代价的那类漂移（R1 / D1）。
 *
 * ## 硬约束：本文件必须**浏览器安全**
 *
 * client 半会 import 它（交权按钮的开关）⇒ **零 import、零 `node:*`**。
 * 需要读文件或碰 SDK 的 EL 参数填法在 `host/el-hub.ts`（node-only）。
 *
 * ## 与资产的关系（2026-10-01 依赖化 + 骨架化之后）
 *
 * 资产已是**可依赖的包** `@mecha/dsh-panel`（`file:` 依赖 ⇒ node_modules 里是 **junction**，
 * 磁盘上只有一份代码）：**EL 不再持有副本、也不再有"参数块文件"**。
 * host 半的建桥/参数装配归**资产骨架**（`mount-host-plugin.ts`）⇒ EL 的值只在这里：
 * 骨架参数（`logLabel` / `clientInfo` / `offlineHint`）与面板参数（`EL_PANEL`，经骨架注入
 * host 半、client 半自己 `configurePanel` 注入一次）；端点**怎么算**在 `host/el-hub.ts`。
 */

/**
 * EL 的插件身份（**唯一来源**）。
 *
 * ⚠ 为什么要有这一项：`name`（Cordis 插件名）/ `serverName`（MCP 工具命名空间
 * `mcp__<serverName>__<tool>`）/ 骨架的 `logLabel`（日志前缀 + 服务名）在 dsh 侧
 * **是同一个字符串**，而它此前在 `src/index.ts` 里被写了十来遍。
 * 那是"同一事实多个家"——**改漏一处就是静默错**（工具命名空间与子注册表名不一致）。
 * ⇒ 现在：`PLUGIN` 一处声明，其余全部引用它（骨架的 `logLabel` 也取 `PLUGIN.name`）。
 */
export const PLUGIN = {
  /** Cordis 插件名（`src/index.ts` / `src/client/index.ts` 的 `export const name`）。 */
  name: 're0-mecha',
  /** MCP 工具命名空间：工具名形如 `mcp__re0-mecha__<tool>`。 */
  serverName: 're0-mecha',
} as const

/** EL 填给共享资产 opt-in 件 / 骨架的值（**唯一来源**）。 */
export const EL_BRIDGE = {
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
 * EL 的面板参数（**注入用**；取代原来的"参数块文件"）。
 *
 * 资产改成依赖包之后，项目**不能再改资产文件**（三仓共用同一份）⇒ 参数改为**启动时注入**：
 * `configurePanel(EL_PANEL)`，读取侧 `panelConfig()` **未注入即抛**（资产不给默认值——
 * 猜一个会把"忘了注入"演成"连上了但面板是空的"）。
 * 两个消费方半（host / client）**各注入一次**（两份 bundle，互不共享内存）。
 */
export const EL_PANEL = {
  /** host 半注册的只读路由；client 半 fetch 同一个串。 */
  routePath: '/re0-mecha/monitor-url',
  /** 权威写的**监控**端口文件名（⚠ 与 `EL_BRIDGE.mcpPortFile` 是两个不同的文件）。 */
  portFile: '.monitor-port',
  /** 面板标题（纯展示；模型不读它）。 */
  title: '机甲监控',
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
