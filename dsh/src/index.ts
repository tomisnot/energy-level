/**
 * re0-mecha dsh 插件 —— **HOST 入口**（Cordis 插件，函数形态）。
 *
 * 职责（AI 侧集成轨）：
 *  1. 建自愈 MCP 桥连到软件（`httpSessionFactory` + `MechaMcpBridge`）——接管 dsh stock HTTP
 *     桥缺的重连（治《DSH-MCP现状》§2.3 的"重启即永久 404"）。
 *  2. 桥发现工具 → `syncTools` 原生注册进 `ctx.tools`（`mcp__re0-mecha__<tool>`）；重连后
 *     工具集变化会再同步。
 *  3. 断联显式信号：状态迁移写日志；软件离线时工具调用抛**可读**离线错误（桥内已实现）。
 *
 * 安全红线：**只 connect、绝不 spawn 权威**（人启动 launcher = 权威，T4）。
 * 依赖方向：本文件 import dsh 的 `@deepseek-ai/*`（peer，宿主提供）+ 本地纯逻辑模块。
 *
 * **client 半（dsh 界面：◈监控 按钮 + 右栏监控面板 + 「切换至 GUI」按钮）随
 * 两模式收拢重建**（2026-09-24）：面板轮询权威 :8767 的只读 /activity；切换按钮
 * 经本文件注册的 /re0-gui command 写 .mode-request，看门人接手进程交接。
 */
import type { Context } from '@deepseek-ai/cordis'
import { MechaMcpBridge } from './host/mcp-bridge.ts'
import { httpSessionFactory } from './host/mcp-session-http.ts'
import { resolveHubUrl, buildRequestInit } from './host/config.ts'
import { syncTools, type ToolDisposers } from './host/register-tools.ts'
import { elRootFromPortFile, writeModeRequest } from './host/mode-request.ts'

/** Cordis 插件显示名（诊断用）。 */
export const name = 're0-mecha'

/** 等待 dsh 的 tools 服务就绪再激活（工具注册需要它）；commands 供 /re0-gui 切换。 */
export const inject = ['tools', 'commands']

/**
 * 插件配置（用户在 cordis.patch.yml 的 entry `config:` 里给；全部可选，缺省在 apply 里兜）。
 *
 * 刻意**不导出 schemastery 的 Config 校验 schema**：那会引入对 `@deepseek-ai/schemastery` 的
 * **运行时** import，而 out-of-tree 插件的 node_modules 里没有它（它是 dsh 宿主提供的 peer）——
 * 宿主是否替 out-of-tree 插件解析 @deepseek-ai/* 尚待活体确认。去掉后，host 半对
 * @deepseek-ai/* **只剩类型 import（编译期擦除）**、运行时零 peer 依赖，加载最稳。
 */
export interface Config {
  /** 显式 Hub MCP 端点；留空则从 mcpPortFile 解析。 */
  hubUrl?: string
  /** 端口文件路径（launcher 写；默认项目根 `.mcp-port`，相对 dsh 的 cwd）。 */
  mcpPortFile?: string
  /** 工具命名空间：`mcp__<serverName>__<tool>`。 */
  serverName?: string
  /** 远程 Hub 鉴权 token（→ Authorization: Bearer）；本地默认空。需求③远程接缝。 */
  hubToken?: string
  /** 额外请求头。 */
  hubHeaders?: Record<string, string>
  /** 重连退避：首次延迟 ms（默 300）。 */
  reconnectInitialDelayMs?: number
  /** 重连退避：上限 ms（默 15000）。 */
  reconnectMaxDelayMs?: number
  /** 重连最多尝试次数（0=无限后台重连）。 */
  reconnectMaxAttempts?: number
}

/** 挂载 host 桥：连 Hub、注册工具、状态信号；teardown 时注销工具并关桥。 */
export async function apply(ctx: Context, config: Config = {}): Promise<void> {
  await ctx.effect(async () => {
    const serverName = config.serverName || 're0-mecha'
    let disposers: ToolDisposers = new Map()
    // 端点**每次建会话现解**（不固化）：权威重启换端口（实测报告 §0，如
    // 56093→52755）后，桥的重连工厂重读 .mcp-port 即连上新端口——老会话
    // 不再静默失效。显式 hubUrl（远程设备）在 resolveHubUrl 里优先级最高，
    // 行为不变。lastUrl 仅供日志展示最近一次解析结果。
    let lastUrl = ''
    const resolveUrl = async (): Promise<string> => {
      lastUrl = await resolveHubUrl(config)
      return lastUrl
    }

    const bridge = new MechaMcpBridge({
      // requestInit：远程 Hub 鉴权头（需求③接缝；本地 hubToken 空 → 不带头）。
      sessionFactory: httpSessionFactory({ url: resolveUrl, requestInit: buildRequestInit(config) }),
      reconnect: {
        initialDelayMs: config.reconnectInitialDelayMs ?? 300,
        maxDelayMs: config.reconnectMaxDelayMs ?? 15000,
        maxAttempts: config.reconnectMaxAttempts ?? 0,
      },
      logger: {
        info: m => ctx.logger?.info?.(m),
        warn: m => ctx.logger?.warn?.(m),
        error: m => ctx.logger?.error?.(m),
      },
      // 桥发现/刷新工具 → 两阶段 swap 重注册（首连与每次重连后都走这里）。
      onTools: tools => {
        disposers = syncTools(ctx, bridge, serverName, tools, disposers)
      },
      // 断联显式信号（报告 §4.3.1）：状态迁移落日志，人/AI 不再面对"工具在但静默坏"。
      onStatus: status => {
        if (status === 'offline') ctx.logger?.warn?.('[re0-mecha] Hub 离线：工具暂不可用，后台重连中')
        else if (status === 'ready') ctx.logger?.info?.(`[re0-mecha] 已连上 Hub MCP：${lastUrl}`)
        else if (status === 'reconnecting') ctx.logger?.warn?.('[re0-mecha] 与 Hub 的连接中断，重连中…')
      },
    })

    ctx.logger?.info?.('[re0-mecha] 挂载 Hub MCP 桥（端点每次建会话现解 .mcp-port，只 connect 不 spawn 权威）')
    await bridge.start()   // Hub 未起也不抛：插件照常加载，桥后台重连，起来后自动注册工具

    // /re0-gui：dsh 界面「切换至 GUI」按钮的落点——写 .mode-request（'gui'），
    // 看门人轮询到就 terminate 权威+dsh、起 GUI 模式。插件只留意图、不 spawn/kill。
    const root = elRootFromPortFile(config.mcpPortFile)
    const disposeCmd = ctx.commands!.register({
      name: 're0-gui',
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
    })

    return async () => {
      try { disposeCmd() } catch { /* ignore */ }
      for (const dispose of disposers.values()) {
        try { dispose() } catch { /* ignore */ }
      }
      await bridge.close()
    }
  }, 're0-mecha: host MCP bridge')
}
