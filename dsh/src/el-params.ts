/**
 * EL 的**参数家**（本项目专有；**不进共享资产指纹表**）。
 *
 * ## 为什么单独立一个文件
 *
 * **资产覆盖不到的项目值**需要一个家：MCP 桥的日志前缀 / 断联指引 / 客户端标识 / 端点默认。
 * 散在 host/client 两半的各调用点上就是两处真值，正是本工程反复付代价的那类漂移（R1 / D1）。
 *
 * ## 硬约束：本文件必须**浏览器安全**
 *
 * client 半会 import 它（那一行 bootstrap 间接依赖）⇒ **零 import、零 `node:*`**。
 * 需要读文件或碰 SDK 的 EL 参数填法在 `host/el-hub.ts`（node-only）。
 *
 * ## 与资产的关系（2026-10-01 依赖化 + 骨架化之后）
 *
 * 资产已是**可依赖的包** `@mecha/dsh-panel`（`file:` 依赖 ⇒ node_modules 里是 **junction**，
 * 磁盘上只有一份代码）：**EL 不再持有副本、也不再有"参数块文件"**。
 * host 半的建桥/参数装配归**资产骨架**（`mount-host-plugin.ts`）⇒ EL 的值只在这里：
 * 骨架参数（`logLabel` / `clientInfo` / `offlineHint`）；端点**怎么算**在 `host/el-hub.ts`。
 *
 * ## ⭐ 2026-10-09：**面板参数从这里删掉了**（那个常量已无消费者）
 *
 * 监控面（路由 / 端口文件名 / 标题）与那条**地址路由**的注册都**上提进框架**了：
 *
 * | 事 | 谁做 | 家在哪 |
 * |---|---|---|
 * | 面板参数（3 项） | 框架生成 → 框架的 host 插件读 **env** | ⭐ **声明**（`energy-level/mecha_site.py` 的 `Service.monitor`） |
 * | 地址路由的注册 + 现取端口 | ⭐ **框架的 host 插件**（`mecha-panel-host.mjs`） | 框架（按 `MECHA_PANEL_PORT_FILES`） |
 * | 右栏 tab 的挂载 | ⭐ **框架的 client 插件**（本仓只留一行 bootstrap） | `dsh/src/client/index.ts` |
 *
 * ⇒ **面板参数常量已删**（留着就是"参数的第二处真值"），`mountHostPlugin` 也**不再传
 *   `panelConfig`**（⭐ **同一条 `routePath` 注册两次是接线错误**）。
 *   ⚠ 判据 `dsh/test/re0-panel.test.ts` 钉着"不许再传 / 不许再加回来"。
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
 * ⚠ **曾经有 `EL_FEATURES.enableSwitchToGui`**（会话头部「→ GUI」按钮 + host 的
 * `/re0-gui` command 的能力开关）。**三层架构（2026-10-08）把它连同整条链删了**：
 * 那条路靠"host 写 `.mode-request`、看门人轮询到就做进程交接"；而切写权现在走
 * **控制端点**（`.control-port`）、**不动进程**、不留意图文件 ⇒ 没有存在理由。
 */

/**
 * ⚠ **已删（2026-10-09）：会话头部标签常量 与 `client/buttons.tsx`。**
 *
 * 那个「◈ 监控」按钮**阶段① 起归框架**（全生态**唯一一个**，语义 = 开右栏）
 * ⇒ `buttons.tsx` 早就**没有任何 import 方**（死文件），而那个标签常量只服务它
 * ⇒ 两个一起删（R14：删除类改动要**全仓搜残留**，别留"指着不存在的东西"的文件与常量）。
 *
 * ⚠ 已删（2026-10 阶段①）：右栏 tab 类型常量与 `headerPlan()` —— 它们服务的是
 * "本仓自己注册会话头部按钮 + 自己注册右栏 tab 类型"那套写法，那两件事归框架
 * （`mountClientPanel`；tab 的 `kind`/`id` 按三层架构 §7.3.3 = `mecha:EL` / `mecha-panel:EL`）。
 */
