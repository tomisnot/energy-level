/**
 * **re0-mecha 项目自有**的面板判据（共享资产之外的那一半）。
 *
 * ⚠ 共享单测（`src/dsh-panel/*.test.ts`）**逐字复制、不许改**；项目要加的"自家语料"
 * 按资产 `dsh-panel/README.md` 的约定**放这里**：项目字面量（路由、端口文件名、
 * 历史默认端口）只许出现在本文件与参数块里。
 *
 * 钉三件事：
 *  1. **参数块就是 EL 的事实**（路由/端口文件名/标题）——它是本仓唯一允许项目化的文件；
 *  2. ⭐ **单一来源**：路由与端口文件名**只写在参数块里**，host/client 两半都**引用**它
 *     （资产 README：「两半同源同包 ⇒ 单一来源在包内自动满足」）⇒ 这里按**内容**读源码，
 *     断言那两个字面量没有在别处再抄一份（抄一份就会漂，这正是本工程反复失手处）；
 *  3. 反面语料：**历史默认端口 `8767` 不许出现在面板源码里**（它早已改成动态发现；
 *     资产共享单测里不许出现项目字面量，所以这条只能放这里）。
 *
 * 跑：`npm test`（同仓其余单测一起）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { PANEL_CONFIG } from '../src/dsh-panel/panel-config.ts'

const HERE = dirname(fileURLToPath(import.meta.url))
const SRC = join(HERE, '..', 'src')
const read = (rel: string): string => readFileSync(join(SRC, rel), 'utf8')

/**
 * ⚠ **不要** import `client/index.ts`（也不 import 组件）：它 import `MonitorTabBody.tsx`
 * → `react`，而 react 是 **dsh 宿主提供的 peer**、本仓 devDeps 里没有 ⇒ 一 import 就
 * `ERR_MODULE_NOT_FOUND`。所以这一半的"单一来源"用**按内容读源码**来断言（判据本来也
 * 该能脱离运行时而成立）。
 */
function literalOutsideParams(literal: string, files: string[]): string[] {
  return files.filter((f) => read(f).includes(literal))
}

test('参数块是本仓的事实：路由 / 端口文件 / 标题', () => {
  assert.equal(PANEL_CONFIG.ROUTE_PATH, '/re0-mecha/monitor-url')
  assert.equal(PANEL_CONFIG.PORT_FILE, '.monitor-port')
  assert.ok(PANEL_CONFIG.TITLE.trim().length > 0, '标题不许空')
  assert.ok(PANEL_CONFIG.PORT_FILE.trim().length > 0,
    'PORT_FILE 留空 ⇒ 路由一律 503（共享资产的刻意设计）——本仓必须填')
})

test('⭐ 单一来源：两半都引用参数块（路由字面量不在别处再抄一份）', () => {
  const client = read('client/index.ts')
  assert.match(client, /ADDRESS_ROUTE\s*=\s*PANEL_CONFIG\.ROUTE_PATH/,
    'client 半必须**引用**参数块的路由，而不是自带字面量')
  assert.match(read('index.ts'), /path:\s*PANEL_CONFIG\.ROUTE_PATH/,
    'host 半注册的路由必须来自参数块')
  assert.match(read('index.ts'), /portFile:\s*PANEL_CONFIG\.PORT_FILE/,
    'host 半读的端口文件名必须来自参数块')
  const offenders = literalOutsideParams(PANEL_CONFIG.ROUTE_PATH,
    ['index.ts', 'client/index.ts', 'client/MonitorTabBody.tsx', 'client/buttons.tsx'])
  assert.deepEqual(offenders, [], `路由字面量被抄进了：${offenders.join('、')}`)
})

test('反面语料：历史默认端口 8767 不在面板源码里（端口早已动态发现）', () => {
  const offenders = literalOutsideParams('8767',
    ['index.ts', 'client/index.ts', 'client/MonitorTabBody.tsx', 'client/buttons.tsx',
      'dsh-panel/panel-config.ts'])
  assert.deepEqual(offenders, [], `面板源码里不该再出现 8767：${offenders.join('、')}`)
})
