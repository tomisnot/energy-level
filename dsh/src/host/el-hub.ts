/**
 * EL 的 **node 侧参数接线**：把 EL 的值（`el-params.ts`）填进共享资产的 opt-in 件。
 *
 * 为什么单独一层而不是就地改资产文件：资产件的纪律是**逐字复制**（`dsh-panel/README.md`
 * 的「复制约定」），项目值只能**经参数**传进去 ⇒ 需要一个 EL 自己的地方来持有"哪几个参数、
 * 填什么值"。本文件就是那个地方（node-only 那一半；浏览器安全那一半在 `el-params.ts`）。
 *
 * ⚠ 本文件**不是**资产件，别把它的内容想成"该跟上游同步"的东西；它随 EL 的部署形态变。
 */
import type { BridgeOptions, SessionFactory } from './../dsh-panel/mcp-bridge.ts'
import type { HubUrlConfig } from './../dsh-panel/config.ts'
import { resolveHubUrl } from './../dsh-panel/config.ts'
import { httpSessionFactory } from './../dsh-panel/mcp-session-http.ts'
import { EL_BRIDGE } from './../el-params.ts'

/** EL 的 dsh 插件配置（用户在 cordis patch 的 entry `config:` 里给）。 */
export interface ElHubConfig {
  /** 显式端点（远程设备用）；给了就不读端口文件。 */
  hubUrl?: string
  /** 端口文件路径；缺省用 EL 的 `.mcp-port`（见 `EL_BRIDGE.mcpPortFile`）。 */
  mcpPortFile?: string
}

/**
 * EL 填给资产 `resolveHubUrl` 的完整参数（**纯函数**，可单测）。
 *
 * 资产的口径是"显式 URL > 端口文件 > 项目默认，**缺省不猜端口**"——所以两个项目值
 * （端口文件名、默认端点）必须由 EL 显式给；这就是本函数存在的全部理由。
 */
export function elHubUrlConfig(config: ElHubConfig = {}): HubUrlConfig {
  return {
    hubUrl: config.hubUrl,
    mcpPortFile: config.mcpPortFile?.trim() || EL_BRIDGE.mcpPortFile,
    defaultUrl: EL_BRIDGE.defaultUrl,
  }
}

/** EL 的端点解析（每次建会话现解 ⇒ 权威重启换端口后下一拍跟上）。 */
export async function elResolveHubUrl(config: ElHubConfig = {}): Promise<string> {
  return resolveHubUrl(elHubUrlConfig(config))
}

/** EL 的会话工厂：EL 的客户端标识 + 可注入的请求头（远程鉴权接缝）。 */
export function elSessionFactory(
  url: string | (() => Promise<string>),
  requestInit?: RequestInit,
): SessionFactory {
  return httpSessionFactory({ url, clientInfo: { ...EL_BRIDGE.clientInfo }, requestInit })
}

/**
 * EL 的桥参数（**唯一来源**：桥的 `logLabel` / `offlineHint` 只在这里填）。
 *
 * ⚠ 别在调用点再手写这两个值：`reconnect.test.ts` 的离线断言正是拿本函数的产物去验
 * "EL 的指引真的进了错误文案"（参数必须**真有消费者**，R17）。
 */
export function elBridgeOptions(
  sessionFactory: SessionFactory,
  overrides: Partial<BridgeOptions> = {},
): BridgeOptions {
  return {
    sessionFactory,
    logLabel: EL_BRIDGE.logLabel,
    offlineHint: EL_BRIDGE.offlineHint,
    ...overrides,
  }
}
