/**
 * 监控端点**地址**：同源只读路由 `/re0-mecha/monitor-url`（host 半）。
 *
 * 为什么需要它：面板（client 半）不能自己读磁盘，而监控端点的端口是**动态分配**的
 * （`src/v2_bind.py` 起端点时 `port=0`，把实际端口写进项目根 `.monitor-port`）。
 * 于是"地址"必须由 host 半转一次：host 读那个裸端口文件 → 本路由回 `{base}` →
 * 面板 fetch 同源路由拿到地址。
 *
 * 三条设计点（照 ML 的形态，但**不引入描述符**——`.mecha-runtime.json` 那套 schema
 * 是第 5 步 `mecha[launcher]` 的事，框架 ADR 已划出边界；这里只读裸端口）：
 *
 * 1. **绝不回落默认端口**：文件不存在 / 内容非数字 / 端口越界 ⇒ 抛错，路由回
 *    `503` + 可读错误。回落会让"权威没起来"显示成"连上了但空面板"——本仓反复
 *    在防的假绿（硬编码 8767 时代的真实故障模式）。
 * 2. **每次调用现读**：端口漂移（权威重启换端口）后**下一拍**就落在新端口上，
 *    不需要重启 dsh。路由每次被 fetch 都重新读文件，本模块不缓存。
 * 3. **纯逻辑 + 可注入路径**：`resolveMonitorBase(root, portFile)` 是纯函数，
 *    判据可在 Node 里直接喂真文件/坏文件（见 `test/monitor-url.test.ts`）。
 */
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { IncomingMessage, ServerResponse } from 'node:http'

/** host 半注册的同源只读路由（client 半用同一常量字符串）。 */
export const MONITOR_ROUTE_PATH = '/re0-mecha/monitor-url'

/** 权威写的端口发现文件（裸端口文本；与 `.mcp-port` 同在项目根）。 */
export const MONITOR_PORT_FILE = '.monitor-port'

/** 监控端点基址的解析选项。 */
export interface MonitorUrlOptions {
  /** EL 项目根（权威与 dsh 的 cwd 同源）。 */
  root?: string
  /** 端口文件名（相对 root）或绝对路径；默认 `.monitor-port`。 */
  portFile?: string
}

/**
 * 解析监控端点基址（`http://127.0.0.1:<port>`）。
 *
 * 任何一步不对都**抛错**（调用方转成 503），**不回落到任何默认端口**。
 */
export function resolveMonitorBase(opts: MonitorUrlOptions = {}): string {
  const root = opts.root && opts.root.trim() ? opts.root.trim() : process.cwd()
  const name = opts.portFile && opts.portFile.trim() ? opts.portFile.trim() : MONITOR_PORT_FILE
  const path = name.includes('/') || name.includes('\\') ? name : join(root, name)
  let raw = ''
  try {
    raw = readFileSync(path, 'utf8').trim()
  } catch (err) {
    throw new Error(
      `读不到监控端口（${path} 不存在）：软件没在 AI 模式跑？(${(err as Error).message})`,
    )
  }
  if (!/^\d+$/.test(raw)) {
    throw new Error(`监控端口文件内容非法：${JSON.stringify(raw)}（${path}）`)
  }
  const port = Number(raw)
  if (!(port > 0 && port <= 65535)) {
    throw new Error(`监控端口越界：${port}（${path}）`)
  }
  return `http://127.0.0.1:${port}`
}

/** 写一份 JSON 响应（handler 拥有完整响应生命周期，故自己收尾）。 */
export function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  const body = Buffer.from(JSON.stringify(payload), 'utf8')
  res.statusCode = status
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  res.setHeader('Content-Length', String(body.length))
  res.setHeader('Cache-Control', 'no-store')
  res.end(body)
}

/** 路由处理：能解析就 `200 {base}`；解析不了就 `503` + 可读错误（**不回落**）。 */
export function makeMonitorUrlHandler(opts: MonitorUrlOptions = {}) {
  return function monitorUrlHandler(_req: IncomingMessage, res: ServerResponse): void {
    try {
      sendJson(res, 200, { base: resolveMonitorBase(opts) })
    } catch (err) {
      sendJson(res, 503, { error: `监控端点未知：${(err as Error).message}` })
    }
  }
}
