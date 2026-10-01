/**
 * 「切换至 GUI」**能力开关**判据（EL 侧）——它守的是"关着时不留死按钮"。
 *
 * ## 为什么需要它
 *
 * 交权按钮依赖"有人在轮询 `.mode-request`"（EL 的看门人 `launcher.py`）。没有看门人的
 * 部署里，按钮按了没人接 ⇒ **必须能关**，且**关着时按钮与 command 都不注册**。
 * 两半（client / host）**读同一个计划**（`headerPlan`）⇒ 不会出现"按钮在、command 不在"
 * （那正是"按了没反应"的成因）。
 *
 * ## 两类断言
 *
 * * **行为**（主）：`headerPlan(true/false)` 两个方向的产物——开关真的改变结果（R17）。
 * * **结构**（副）：client/host 两半**都不许手抄** id/command（否则开关会与另一半脱钩），
 *   按钮文案只能取 `EL_HEADER_LABEL`（一处真值）。这两条探的是"有人把手写分支加回来"
 *   那一刻——那时行为断言仍绿，但它已经埋了脱钩的种子。
 *
 * 跑：`node --test test/switch-to-gui.test.ts`（**裸跑**）
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { headerPlan } from '../src/el-params.ts'

const SRC = join(dirname(fileURLToPath(import.meta.url)), '..', 'src')
const read = (rel: string): string => readFileSync(join(SRC, rel), 'utf8')

/** 只留代码：去掉块注释与行注释（`//` 前带 `:` 的 http:// 不算注释）。 */
function code(rel: string): string {
  return read(rel)
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1')
}

test('开关开：头部动作 = 监控按钮 + 交权按钮（含它要的 command）', () => {
  const plan = headerPlan(true)
  assert.equal(plan.cockpit.id, 're0-cockpit')
  assert.equal(plan.cockpit.order, 200)
  assert.ok(plan.cockpit.label.trim().length > 0, '按钮标签不许空')
  const sw = plan.switchToGui
  assert.ok(sw, '开着时必须有交权动作')
  assert.equal(sw.id, 're0-to-gui')
  assert.equal(sw.order, 201)
  assert.equal(sw.command, 're0-gui', 'host 半注册的 command 名来自**同一个**计划')
})

test('⭐ 开关关：交权动作是 null ⇒ 两半都不注册按钮与 command（不留死按钮）', () => {
  const plan = headerPlan(false)
  assert.notEqual(plan.cockpit, null, '监控按钮与开关无关，永远在')
  assert.equal(plan.switchToGui, null)
})

test('⭐ 结构性：交权分支只由计划驱动（两个方向都验：没字面量 **且** 真读了计划）', () => {
  const client = code('client/index.ts')
  const host = code('index.ts')
  const buttons = code('client/buttons.tsx')
  for (const [name, src] of [['client/index.ts', client], ['index.ts', host]] as const) {
    for (const lit of ['re0-to-gui', 're0-gui']) {
      assert.ok(!src.includes(lit),
        `${name} 的**代码**里出现了 ${lit} 字面量 ⇒ 开关与另一半会脱钩（按钮在、command 不在）`)
    }
    assert.match(src, /headerPlan\(/, `${name} 必须从 headerPlan 取计划`)
  }
  // 标签唯一来源：组件不许再手抄文案（否则改一处、另一处漂）
  for (const lit of ['◈ 监控', '→ GUI']) {
    assert.ok(!buttons.includes(lit), `buttons.tsx 手抄了标签 ${lit}（应取 EL_HEADER_LABEL）`)
  }
  assert.match(buttons, /EL_HEADER_LABEL/)
})
