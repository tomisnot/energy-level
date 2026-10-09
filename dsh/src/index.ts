/**
 * re0-mecha dsh 插件 —— **HOST 入口**（Cordis 插件，函数形态）。
 *
 * ## 机制全在**资产骨架**里（2026-10-01 上提）
 *
 * `@mecha/dsh-panel/host/mount-host-plugin.ts` 的 `mountHostPlugin(ctx, opts)` 接管了此前本文件
 * 手写的那约 100 行通用机制：会话工厂（含 **SDK 懒加载**）、建桥、logger 转发、
 * `onTools → syncTools` 两阶段 swap、`onStatus` 三态日志、监控地址路由的注册**与注销**、
 * 收尾顺序、面板参数注入。⇒ **本文件只剩 EL 的参数值与自身分支**。
 *
 * 骨架有什么 / EL 给什么：见该模块 docstring 的表；本文件只填值 + 交一个可选 command。
 *
 * 安全红线：**只 connect、绝不 spawn 权威**（框架起软件 = 那个进程）。
 *
 * ⚠ **曾经交一个可选 command**（`/re0-gui`：写 `.mode-request` 请看门人做进程交接）。
 * **三层架构（2026-10-08）删了整条链**：切写权现在走**控制端点**（`.control-port`）、
 * **不动进程** ⇒ "写文件请某某接手"那类东西没有存在理由，本插件**不再交任何 command**。
 *
 * **资产是依赖包**：`@mecha/dsh-panel`（`file:` 依赖 ⇒ node_modules 里是 **junction**，
 * 磁盘上只有一份代码）；面板参数经 `panelConfig` 交给骨架注入（未注入时资产会**抛**）。
 */
import type { Context } from '@deepseek-ai/cordis'
import { buildRequestInit } from '@mecha/dsh-panel/host/config.ts'
import { hostPluginInject, mountHostPlugin } from '@mecha/dsh-panel/host/mount-host-plugin.ts'
import { elResolveHubUrl, elRootFromPortFile } from './host/el-hub.ts'
import { EL_BRIDGE, PLUGIN } from './el-params.ts'

/** Cordis 插件显示名（诊断用）。**取自参数家**（`PLUGIN.name` 是唯一来源）。 */
export const name = PLUGIN.name

/**
 * 顶层 `inject`：骨架恒要 `tools`（+ `webServer` 供监控地址路由）。
 * ⚠ 原先还按交权开关另要 `commands`（注册 `/re0-gui`）—— 那条链已删 ⇒ 不再声明它（R17）。
 */
export const inject = [...hostPluginInject()]

/** 插件配置（cordis patch 的 entry `config:` 里给；全部可选）。 */
export interface Config {
  /** 显式 Hub MCP 端点；留空则从 mcpPortFile 解析。 */
  hubUrl?: string
  /** MCP 端口文件路径（框架起软件时由软件写；缺省项目根 `.mcp-port`）。 */
  mcpPortFile?: string
  /** 工具命名空间 `mcp__<serverName>__<tool>`；缺省 = `PLUGIN.serverName`。 */
  serverName?: string
  /** 远程 Hub 鉴权 token（→ Authorization: Bearer）；本地默认空。需求③远程接缝。 */
  hubToken?: string
  /** 额外请求头。 */
  hubHeaders?: Record<string, string>
  /** 重连退避：首次延迟 ms（**不给就用桥自己的默认**）。 */
  reconnectInitialDelayMs?: number
  /** 重连退避：上限 ms。 */
  reconnectMaxDelayMs?: number
  /** 重连最多尝试次数（0=无限后台重连）。 */
  reconnectMaxAttempts?: number
}

/**
 * 挂载 host 半：**机制归骨架**，这里只填 EL 的值 + 交一个可选 command。
 *
 * ⚠ `logLabel` **同时就是服务名**（骨架特意把两者合并——拆开会把同一个词写两遍）。
 */
export async function apply(ctx: Context, config: Config = {}): Promise<void> {
  // 项目根（`.mcp-port` / `.control-port` / 监控端口文件都在这里）。
  const root = elRootFromPortFile(config.mcpPortFile)

  await mountHostPlugin(ctx, {
    serverName: config.serverName || PLUGIN.serverName,
    logLabel: PLUGIN.name,
    projectRoot: root,
    // 端点怎么算是**项目知识**（显式 URL > 端口文件 > 项目默认），骨架不猜；
    // 它只保存函数引用、每次建会话现解（端口漂移自愈）。
    resolveUrl: () => elResolveHubUrl(config),
    clientInfo: { ...EL_BRIDGE.clientInfo },
    offlineHint: EL_BRIDGE.offlineHint,
    // 远程 Hub 鉴权头（需求③接缝；本地 hubToken 空 ⇒ 不带头）。
    requestInit: buildRequestInit(config),
    // 三个旋钮**只在部署显式给了值时才覆盖**（不给 ⇒ 桥自己的默认；不手抄第二处真值）。
    reconnect: {
      initialDelayMs: config.reconnectInitialDelayMs,
      maxDelayMs: config.reconnectMaxDelayMs,
      maxAttempts: config.reconnectMaxAttempts,
    },
    // ⭐ **不传 `panelConfig`**（2026-10-09）：监控地址路由**上提进框架**了
    //    （框架的 host 插件按声明 + `MECHA_PANEL_PORT_FILES` 现取端口）
    //    ⇒ ⚠ **同一条 `routePath` 注册两次是接线错误**，本仓不再注册它。
    //    ⚠ 这也让参数只剩**一个家**：`energy-level/mecha_site.py` 的 `Service.monitor`
    //      （本仓的面板参数常量已随之删除）。
    // ⚠ **不传 `registerCommand`**：那条"写 .mode-request 请看门人接手"的路已随三层架构删除。
    // 切写权改由**控制端点**（人侧/框架）做 ⇒ 本插件不再交任何 command（也不留死按钮）。
  })
}
