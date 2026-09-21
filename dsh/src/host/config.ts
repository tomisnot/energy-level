/**
 * Hub 端点解析（纯逻辑，无 dsh 依赖，可单测）。
 *
 * 优先级：显式 `hubUrl` > 读 `.mcp-port` 端口文件（人启动 launcher 时写；见 re0-mecha SKILL）
 * > 默认 8765。端口文件格式容错：裸端口 / 完整 URL / 含数字的任意串。
 */
import { readFile } from 'node:fs/promises'

export interface HubUrlConfig {
  hubUrl?: string
  mcpPortFile?: string
}

const DEFAULT_URL = 'http://127.0.0.1:8765/mcp'

/** 解析 Hub 的 MCP 端点 URL。绝不抛（解析不到就回落默认，交由桥的重连去处理"连不上"）。 */
export async function resolveHubUrl(config: HubUrlConfig): Promise<string> {
  const explicit = config.hubUrl?.trim()
  if (explicit) return explicit
  const portFile = config.mcpPortFile?.trim() || '.mcp-port'
  try {
    const raw = (await readFile(portFile, 'utf8')).trim()
    if (/^https?:\/\//i.test(raw)) return raw
    const m = raw.match(/(\d{2,5})/)
    if (m && m[1]) return `http://127.0.0.1:${m[1]}/mcp`
  } catch {
    /* 没有端口文件（Hub 未起 / cwd 不对）→ 回落默认，桥会后台重连 */
  }
  return DEFAULT_URL
}

export interface RemoteConfig {
  /** 远程 Hub 的鉴权 token（→ `Authorization: Bearer`）。本地默认空=不带。 */
  hubToken?: string
  /** 额外请求头（自定义鉴权/追踪）。 */
  hubHeaders?: Record<string, string>
}

/**
 * 由远程配置构造 fetch 的 `RequestInit`（鉴权头）。纯函数、可单测。
 *
 * 无 token/头 → `undefined`（本地默认，不带头）。这是"远程跨设备"（需求③）的插件侧
 * 接缝：将来 Hub 的写通道若经 L3 双闸安全地远程暴露（T6 今天写通道仍只 localhost），
 * 插件带上 token 即可连远程 Hub；`cockpitUrl` 同理可指向远程 cockpit-server（读通道
 * 早已支持 `--host 0.0.0.0 --token`）。
 */
export function buildRequestInit(config: RemoteConfig): RequestInit | undefined {
  const headers: Record<string, string> = { ...(config.hubHeaders ?? {}) }
  if (config.hubToken) headers.Authorization = `Bearer ${config.hubToken}`
  return Object.keys(headers).length > 0 ? { headers } : undefined
}
