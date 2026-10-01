/**
 * **re0-mecha 项目自有**的面板判据（共享资产之外的那一半）。
 *
 * ⚠ 共享单测（`src/dsh-panel/*.test.ts`）**逐字复制、不许改**；项目要加的"自家语料"
 * 按资产 `dsh-panel/README.md` 的约定**放这里**：项目字面量（路由、端口文件名、
 * 历史默认端口）只许出现在本文件与参数块里。
 *
 * ## 2026-10-01 收窄：**只留资产测不到的那四条**
 *
 * 原先这里还重写了四条"资产自带单测已经覆盖"的用例（本文件是资产测试的前身）。资产
 * 6 个 `.test.ts`（55 例）随资产发货、且被 `checks/dsh_panel_asset_selfcheck.py` 的 C 组
 * 真跑 ⇒ EL 再写一遍是重复。删掉的四条与它们的**新家**：
 *
 * | 删掉的用例 | 现在由谁守 |
 * |---|---|
 * | ① 缺端口文件 ⇒ 503 且**失败体没有 `base`** | `dsh-panel/monitor-url.test.ts:144-151` |
 * | ② 每个非 ok 档都有非空文案、`ok` 档必须真喂数据（R8 对偶） | `dsh-panel/panel-data.test.ts:43-79` |
 * | ③ 地址拿不到 ⇒ **一次都不取数**（不拿空 base 拼相对路径） | `dsh-panel/panel-data.test.ts:157` 起 |
 * | ④ 「飞行记录仪」有事件时画事件行、空表才画"暂无写事件" | `dsh-panel/panel-view.test.ts:45-66` |
 * （④ 那条 EL 当初报的 bug——`payloads` 按请求路径索引导致"永远暂无写事件"——资产单测
 * `panel-data.test.ts:9` 已把该根因写进它的 docstring。）
 *
 * 留下的四条**资产测不到**（都关于"EL 自己的取值/接线"）：
 *  1. **参数块就是 EL 的事实**（路由 / 端口文件名 / 标题）；
 *  2. ⭐ **单一来源**：路由只写在参数块里，host/client 两半都**引用**它；
 *  3. **反面语料**：历史默认端口 `8767` 不许出现在面板源码里；
 *  4. ⭐ **EL client 半的 import 闭包里没有 `node:*`**——资产的闭包检查从**资产自己的
 *     入口**出发，EL 的入口（`client/index.ts` + `buttons.tsx`）它走不到。
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

function literalOutsideParams(literal: string, files: string[]): string[] {
  return files.filter((f) => read(f).includes(literal))
}

// ---------------------------------------------------------------- 1 参数块
test('参数块是本仓的事实：路由 / 端口文件 / 标题', () => {
  assert.equal(PANEL_CONFIG.ROUTE_PATH, '/re0-mecha/monitor-url')
  assert.equal(PANEL_CONFIG.PORT_FILE, '.monitor-port')
  assert.ok(PANEL_CONFIG.TITLE.trim().length > 0, '标题不许空')
  assert.ok(PANEL_CONFIG.PORT_FILE.trim().length > 0,
    'PORT_FILE 留空 ⇒ 路由一律 503（共享资产的刻意设计）——本仓必须填')
})

// ---------------------------------------------------------------- 4 浏览器安全（结构）
/**
 * 沿**相对 import** 走一遍 EL client 半入口的导入闭包（技法照资产
 * `monitor-client.test.ts` 那条，但**根换成 EL 自己的入口**——资产那条走不到这里）。
 */
function closureOf(entry: string): { seen: string[]; nodeHits: string[] } {
  const strip = (src: string) => src
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1')
  const seen = new Set<string>()
  const nodeHits: string[] = []
  const queue = [entry]
  while (queue.length) {
    const rel = queue.shift() as string
    if (seen.has(rel)) continue
    seen.add(rel)
    const isPanel = rel.startsWith('dsh-panel/')
    const abs = join(SRC, isPanel ? 'dsh-panel' : 'client',
      isPanel ? rel.slice('dsh-panel/'.length) : rel)
    for (const line of strip(readFileSync(abs, 'utf8')).split('\n')) {
      if (/^\s*import\s+type\b/.test(line)) continue   // 类型导入会被剥掉，不是运行时依赖
      const m = line.match(/from\s+'([^']+)'/)
      if (!m) continue
      const spec = m[1] ?? ''
      if (spec.startsWith('node:')) { nodeHits.push(`${rel} → ${spec}`); continue }
      if (!spec.startsWith('.')) continue              // 裸包名（react 等）由宿主提供
      const parts = isPanel ? ['dsh-panel', ...rel.slice('dsh-panel/'.length).split('/')]
        : ['client', ...rel.split('/')]
      parts.pop()                                      // 去掉文件名 → 当前目录
      for (const seg of spec.split('/')) {
        if (seg === '.') continue
        else if (seg === '..') parts.pop()
        else parts.push(seg)
      }
      queue.push(parts.join('/'))
    }
  }
  return { seen: [...seen], nodeHits }
}

test('⭐ EL client 半的 import 闭包里没有 node:*（浏览器安全靠结构，不靠树摇）', () => {
  const { seen, nodeHits } = closureOf('dsh-panel/MonitorTabBody.tsx')
  assert.deepEqual(nodeHits, [],
    `client 半闭包里出现 node 内建：${nodeHits.join('、')}`
    + ' ⇒ 浏览器 bundle 会失败（node-only 的实现必须留在 dsh-panel/monitor-url.ts）')
  // R8 自证：闭包必须**真的走过了几个文件**，否则"没命中"只是"什么都没读到"
  assert.ok(seen.length >= 3, `闭包只走了 ${seen.length} 个文件 ⇒ 检查可能没生效：${seen}`)
  assert.ok(seen.includes('dsh-panel/routes.ts'), `闭包里应当有 routes.ts：${seen}`)
  assert.ok(!seen.includes('dsh-panel/monitor-url.ts'),
    `client 半不许碰到 node-only 的 monitor-url.ts：${seen}`)
  // 对偶（不许空洞）：node 半**必须**仍然依赖 node——否则上一条"没东西可抓"
  assert.match(read('dsh-panel/monitor-url.ts'), /from 'node:fs'/,
    'node 半应当 import node:fs（否则本检查没有可抓之物）')
})

// ---------------------------------------------------------------- 2 单一来源
test('⭐ 单一来源：两半都引用参数块（路由字面量不在别处再抄一份）', () => {
  assert.match(read('index.ts'), /path:\s*PANEL_CONFIG\.ROUTE_PATH/,
    'host 半注册的路由必须来自参数块')
  assert.match(read('index.ts'), /portFile:\s*PANEL_CONFIG\.PORT_FILE/,
    'host 半读的端口文件名必须来自参数块')
  assert.match(read('dsh-panel/panel-data.ts'), /fetchMonitorBase\(doFetch,\s*PANEL_CONFIG\.ROUTE_PATH\)/,
    '面板取址必须用参数块的路由')
  const offenders = literalOutsideParams(PANEL_CONFIG.ROUTE_PATH,
    ['index.ts', 'client/index.ts', 'client/buttons.tsx', 'dsh-panel/MonitorTabBody.tsx'])
  assert.deepEqual(offenders, [], `路由字面量被抄进了：${offenders.join('、')}`)
})

// ---------------------------------------------------------------- 3 反面语料
test('反面语料：历史默认端口 8767 不在面板源码里（端口早已动态发现）', () => {
  const offenders = literalOutsideParams('8767',
    ['index.ts', 'client/index.ts', 'client/buttons.tsx', 'dsh-panel/MonitorTabBody.tsx',
      'dsh-panel/panel-config.ts'])
  assert.deepEqual(offenders, [], `面板源码里不该再出现 8767：${offenders.join('、')}`)
})
