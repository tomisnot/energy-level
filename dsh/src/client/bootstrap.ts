/**
 * 读取 host 经 `webserver/index-inject` 注入的 bootstrap（`globalThis.__RE0_MECHA__`）。
 *
 * host 半在 web 模式把 `{cockpitUrl}` 注进 index.html；client 面板据此 iframe cockpit-server。
 * 非 web 模式 / host 半未加载时返回 null（面板给出可读提示，不崩）。
 */
export interface Re0Bootstrap {
  /** cockpit-server 只读 web UI 地址。 */
  cockpitUrl?: string
  /** Hub 健康（可选，host 后续可注入）。 */
  hubHealthy?: boolean
}

export function readBootstrap(): Re0Bootstrap | null {
  const raw = (globalThis as { __RE0_MECHA__?: unknown }).__RE0_MECHA__
  if (!raw || typeof raw !== 'object') return null
  return raw as Re0Bootstrap
}
