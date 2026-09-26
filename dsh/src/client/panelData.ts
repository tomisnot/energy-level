/**
 * 面板**数据层**（无 react、无 DOM）：取址 → 取数 → 归约成**唯一状态** → 文案/徽标映射。
 *
 * ## 为什么把它从组件里抽出来
 *
 * 1. **判据要落在消费者这一层**：抽出来后 Node 能直接喂真形状/坏形状（不需要 react——
 *    react 是 dsh 宿主提供的 peer，本仓 devDeps 里没有）；
 * 2. **"面板不空白"要成为可测的性质**：状态 → 文案的映射是纯函数，判据能**逐个分支**
 *    断言"有可读文本"；组件只负责把它塞进 DOM（`dangerouslySetInnerHTML`）。
 *
 * ## 取数全走 mecha 共享资产
 *
 * `fetchMonitorBase` / `readJson` / `panelState` 全部来自逐字复制的
 * `../dsh-panel/monitor-client.ts`（提交态 `58e2f8a`）：
 * * **三档错误态**（`address` 地址拿不到 / `offline` 基础路由失败 / `route` 附加路由失败）
 *   分开——混在一起会让真因被"权威离线"吃掉；
 * * **`empty` 档**：连上了、也 200、但一点数据都没有 ⇒ 单列一档（没有它，"取数成功但
 *   面板空白"会伪装成成功）；
 * * **零回落**：地址拿不到就是拿不到，绝不返回空地址/相对路径去 fetch。
 */
import {
  displayTier,
  fetchMonitorBase,
  isNonDegenerate,
  panelState,
  readJson,
  statsLine,
  type PanelState,
} from '../dsh-panel/monitor-client.ts'
import { PANEL_CONFIG } from '../dsh-panel/panel-config.ts'

/**
 * 路由与请求路径。
 *
 * ⚠ **键必须用 `BASIC_ROUTES` 的原样成员**（`/history`、`/config`）：共享资产的
 * `panelState` 是拿**键**去索引 payloads 再喂 `monitorStats` 的（`payloads['/history']`）。
 * 若把键写成带查询串的 `/history?since_seq=0`，事件数会**恒为 0**（实测踩过：自证行
 * 打出"事件行=0"而数据其实到了）——查询串只属于**请求路径**，不属于键。
 */
export const HISTORY_ROUTE = '/history'
export const CONFIG_ROUTE = '/config'
/** 实际请求路径：全量尾（`since_seq=0`），键仍用 `HISTORY_ROUTE`。 */
export const HISTORY_PATH = '/history?since_seq=0'
export const CONFIG_PATH = CONFIG_ROUTE

/**
 * 一拍取数：地址 → 两条基础路由 → 归约。
 *
 * ⚠ **地址拿不到时不再 fetch 任何东西**（判据②/③ 守这条）：不拿空 base 去拼相对路径，
 * 否则"软件没在 AI 模式跑"会被显示成"连上了但面板空白"。
 */
export async function loadPanel(doFetch: typeof fetch = fetch): Promise<PanelState> {
  const base = await fetchMonitorBase(doFetch, PANEL_CONFIG.ROUTE_PATH)
  if (base.ok !== true) return panelState({ base })
  const [history, config] = [
    await readJson(base.data, HISTORY_PATH, doFetch),
    await readJson(base.data, CONFIG_PATH, doFetch),
  ]
  return panelState({ base, basic: { [HISTORY_ROUTE]: history, [CONFIG_ROUTE]: config } })
}

/** 状态 → **可读文案**（`ok` 档返回空串：那一档要画数据，不画文案）。 */
export function stateText(state: PanelState): string {
  switch (state.kind) {
    case 'ok':
      return ''
    case 'address':
    case 'offline':
      // 显示档位把 address 显式塌缩成 offline（共享资产的 `displayTier`）：两者对人
      // 是同一件事——"权威没起来"；但**档位本身**仍分开（日志/判据里能辨真因）。
      return `连不上权威（软件没在 AI 模式跑？）：${state.detail}`
    case 'route':
      return `权威在，但附加路由失败：${state.detail}`
    case 'empty':
      return `连上了但没拿到数据（${state.detail}）`
    default:
      return assertNeverText(state)
  }
}

/** 状态 → 徽标（类名 + 文案）。`null` = 首拍还没回来。 */
export function stateBadge(state: PanelState | null): { cls: string; text: string } {
  if (!state) return { cls: 'ok', text: '连接中…' }
  switch (state.kind) {
    case 'ok': {
      const mode = modeOf(state)
      return {
        cls: mode === 'ai' ? 'ai' : mode === 'locked' ? 'warn' : 'ok',
        text: mode === 'ai' ? '写权 AI' : mode === 'locked' ? '全员禁写'
          : mode === 'human' ? '写权在人' : '连接中…',
      }
    }
    case 'address':
    case 'offline':
      return { cls: 'error', text: displayTier(state.kind) === 'offline' ? '权威离线' : '地址未知' }
    case 'route':
      return { cls: 'warn', text: '路由异常' }
    case 'empty':
      return { cls: 'warn', text: '连上但无数据' }
    default:
      return assertNeverBadge(state)
  }
}

/** `ok` 档里取写权模式（v1 的面板就是拿 `mode` 做徽标）。 */
export function modeOf(state: PanelState): string {
  if (state.kind !== 'ok') return ''
  const h = state.payloads[HISTORY_ROUTE] as { mode?: unknown } | undefined
  return typeof h?.mode === 'string' ? h.mode : ''
}

/** R8 自证行（**判据/日志必须把它打出来**，否则"非退化"只是声称）。 */
export function selfCheckLine(state: PanelState): string {
  if (state.kind !== 'ok') return stateText(state)
  return statsLine(state.stats)
}

/** `ok` 档是否非退化（共享资产的判据；`empty` 档就是它的反面）。 */
export function okIsNonDegenerate(state: PanelState): boolean {
  return state.kind === 'ok' && isNonDegenerate(state.stats)
}

/** 兜底：`switch` 到不了的分支**抛**，绝不"什么都不画"（共享资产同款兜底）。 */
function assertNeverText(x: never): never {
  throw new Error(`未处理的 PanelState（文案）：${JSON.stringify(x)}`)
}

function assertNeverBadge(x: never): never {
  throw new Error(`未处理的 PanelState（徽标）：${JSON.stringify(x)}`)
}
