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
 * 安全红线：**只 connect、绝不 spawn 权威**（人启动 launcher = 权威，T4）。
 * 「切换至 GUI」按钮经 `registerCommand` 口子注册的 `/re0-gui` 写 `.mode-request`，
 * 看门人接手进程交接（⚠ `mode-request` 那类"写文件请某某接手"**刻意不进骨架**：
 * 它依赖"有人在轮询"，只有有看门人的项目才有）。
 *
 * **资产是依赖包**：`@mecha/dsh-panel`（`file:` 依赖 ⇒ node_modules 里是 **junction**，
 * 磁盘上只有一份代码）；面板参数经 `panelConfig` 交给骨架注入（未注入时资产会**抛**）。
 */
import type { Context } from '@deepseek-ai/cordis'
import { buildRequestInit } from '@mecha/dsh-panel/host/config.ts'
import {
  hostPluginInject, mountHostPlugin, type HostContext,
} from '@mecha/dsh-panel/host/mount-host-plugin.ts'
import { elResolveHubUrl } from './host/el-hub.ts'
import { elRootFromPortFile, writeModeRequest } from './host/mode-request.ts'
import { EL_BRIDGE, EL_FEATURES, EL_PANEL, PLUGIN, headerPlan } from './el-params.ts'

/** Cordis 插件显示名（诊断用）。**取自参数家**（`PLUGIN.name` 是唯一来源）。 */
export const name = PLUGIN.name

/**
 * 顶层 `inject`：骨架恒要 `tools`（+ `webServer` 供监控地址路由）；交权开关开着时另要
 * `commands`（注册 `/re0-gui`）⇒ 关着时不声明用不到的服务（R17）。
 */
export const inject = [
  ...hostPluginInject(),
  ...(headerPlan(EL_FEATURES.enableSwitchToGui).switchToGui ? ['commands'] : []),
]

/** 插件配置（cordis patch 的 entry `config:` 里给；全部可选）。 */
export interface Config {
  /** 显式 Hub MCP 端点；留空则从 mcpPortFile 解析。 */
  hubUrl?: string
  /** MCP 端口文件路径（launcher 写；缺省项目根 `.mcp-port`）。 */
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
  const switchAction = headerPlan(EL_FEATURES.enableSwitchToGui).switchToGui
  // 项目根（`.mcp-port` 与 `.mode-request` 同址；监控端口文件也在这里）。
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
    panelConfig: EL_PANEL,
    // ⚠ **能力开关**（`EL_FEATURES.enableSwitchToGui`）：它依赖"有人在轮询 `.mode-request`"
    // （EL 的看门人）；**关着时连这个口子都不传** ⇒ 骨架什么都不注册（不留死按钮/死缝）。
    // 按钮那侧由 client 半按**同一个** `headerPlan` 同步开关。
    ...(switchAction === null ? {} : {
      registerCommand: (host: HostContext) => host.commands?.register({
        name: switchAction.command,
        description: '把控制权交还人类：请求看门人切换到 GUI 模式（本 dsh 实例将退出）',
        handler: async () => {
          try {
            const path = await writeModeRequest(root, 'gui')
            return { kind: 'success' as const,
              text: `已请求切换至 GUI 模式（${path}）。本 dsh 实例即将随交接退出。` }
          } catch (e) {
            return { kind: 'error' as const, text: `写 .mode-request 失败：${(e as Error).message}` }
          }
        },
      }) ?? null,
    }),
  })
}
