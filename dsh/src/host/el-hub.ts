/**
 * EL 的 **node 侧参数接线**：把 EL 的端点知识填进资产包的解析规则。
 *
 * ## 2026-10-01 瘦身（host 半骨架上提之后）
 *
 * 建桥 / 会话工厂 / 桥参数（`logLabel` / `offlineHint`）/ 工具注册 / 监控路由 / 收尾
 * 现在是资产骨架 `mount-host-plugin.ts` 的职责 ⇒ 原先在这里的 `elSessionFactory` 与
 * `elBridgeOptions` **已删**（留着就是"两处都懂怎么建桥"）。
 *
 * ⇒ 本文件只剩**项目知识**：**端点怎么算**。资产拥有"显式 URL > 端口文件 > 项目默认"这条
 * **规则**，EL 拥有那两个**名字**（端口文件名 / 默认端点）。
 *
 * ⚠ 本文件不是资产件；它随 EL 的部署形态变（比如哪天默认端点变了）。
 */
import type { HubUrlConfig } from '@mecha/dsh-panel/host/config.ts'
import { resolveHubUrl } from '@mecha/dsh-panel/host/config.ts'
import { dirname, join } from 'node:path'
import { EL_BRIDGE } from './../el-params.ts'

/**
 * 由 `.mcp-port` 路径推 EL 根（端口文件所在目录）；缺省回落 cwd。
 *
 * ⚠ 它原来住在 `host/mode-request.ts`（那个文件写"切换意图" `.mode-request`）。
 * **三层架构（2026-10-08）废掉了那条路**：切写权走**控制端点**（`.control-port`）、
 * **不动进程**，所以"写文件请某某接手"没有存在理由 ⇒ 那个文件整个删了，
 * **但本函数还有消费者**（`index.ts` 拿它算 EL 根）⇒ 搬到此处。
 */
export function elRootFromPortFile(mcpPortFile?: string): string {
  const pf = mcpPortFile?.trim() || '.mcp-port'
  return dirname(join(process.cwd(), pf))
}

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

/** EL 的端点解析（骨架把它包成"每次建会话现解" ⇒ 权威重启换端口后下一拍跟上）。 */
export async function elResolveHubUrl(config: ElHubConfig = {}): Promise<string> {
  return resolveHubUrl(elHubUrlConfig(config))
}
