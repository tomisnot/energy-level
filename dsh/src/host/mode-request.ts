/**
 * 写"切换意图"文件（.mode-request）——纯逻辑，无 dsh 依赖，可单测。
 *
 * dsh 界面里「切换至 GUI」按钮经 /re0-gui command 走到 host，host 写这个文件；
 * 看门人（EL 根 launcher.py）轮询到就 terminate 权威+dsh、起 GUI 模式。**插件只
 * 留意图、不自己 spawn/kill**（编排归看门人，安全红线同 host：只 connect 不 spawn）。
 *
 * 落点 = mcpPortFile 所在目录（权威与看门人同源，默认项目根 `.mcp-port` 的 dirname）。
 */
import { writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

/** 由 .mcp-port 路径推 EL 根（端口文件的目录）；缺省回落 cwd。 */
export function elRootFromPortFile(mcpPortFile?: string): string {
  const pf = mcpPortFile?.trim() || '.mcp-port'
  return dirname(join(process.cwd(), pf))
}

/** 写 .mode-request（内容 'gui'）。返回写入的绝对路径。绝不吞错——写失败让 command 如实报错。 */
export async function writeModeRequest(root: string, mode = 'gui'): Promise<string> {
  const path = join(root, '.mode-request')
  await writeFile(path, mode, 'utf8')
  return path
}
