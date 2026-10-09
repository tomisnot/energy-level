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
  offlineHint: '请确认软件已起来（人经统一启动器 python -m mecha --all 起，或直接 app_entry.py --mcp）；工具仍在列表但暂不可用。',
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
 * ⚠ **曾经有 `EL_FEATURES.enableSwitchToGui`**（会话头部「→ GUI」按钮 + host 的
 * `/re0-gui` command 的能力开关）。**三层架构（2026-10-08）把它连同整条链删了**：
 * 那条路靠"host 写 `.mode-request`、看门人轮询到就做进程交接"；而切写权现在走
 * **控制端点**（`.control-port`）、**不动进程**、不留意图文件 ⇒ 没有存在理由。
 */

/** 会话头部动作的**标签**（唯一来源：`buttons.tsx` 与槽位注册都用它，不许手抄两遍）。 */
export const EL_HEADER_LABEL = {
  cockpit: '◈ 监控',
} as const

/**
 * ⚠ **已删（2026-10 阶段①）：`EL_TAB` 与 `headerPlan()`。**
 *
 * 它们服务的是"本仓自己注册会话头部按钮 + 自己注册右栏 tab 类型"那套写法。
 * 阶段① 起那两件事**归框架**（`mountClientPanel`：按钮**唯一一个**、语义 = 开右栏；
 * tab 的 `kind`/`id` 按三层架构 §7.3.3 = `mecha:EL` / `mecha-panel:EL`）
 * ⇒ 本仓不再需要它们：`EL_TAB` **已无消费者**，而 `headerPlan()` 的 `order`/`label`
 * 是框架那份按钮的事（本仓再留一份就是**第二份真值**）。
 *
 * ⚠ 顺带修掉一句**过期注释**：原先那行写着"dsh 的 `sidebarRightTabs` 与本仓的 slot key
 * 必须同串" —— 那是本仓自己注册 slot 时的约束；现在 slot key 由框架用**它自己那个
 * `id`**（`mecha-panel:EL`）负责，本仓不再拼 slot key。
 */
