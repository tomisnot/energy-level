/**
 * 驾驶舱主界面模式的极小模块级 store。
 *
 * 为什么要模块级：切换按钮在 `conversation.session.header.actions`（scope=session），
 * 驾驶舱覆盖层在 `shell.overlay`（scope=root）——slot 的 store 是逐注册绑定的，跨这两个
 * 不同 scope 的组件共享不了，故用一个模块级单例 observable 桥接（配 useSyncExternalStore）。
 *
 * 副作用：翻转时给 `<html>` 打/去 `data-re0-cockpit="on"`（注入 CSS 的总开关），并按稳定
 * 锚点给 dsh 的三列打 `data-re0-*` 标记——列类名是 CSS Modules hash、CSS 选不中，只能运行时
 * 靠 `[data-rightbar-col]`/`[data-shell-overlay]` 的兄弟关系定位后打标。
 */

type Listener = () => void

let cockpitMode = false
const listeners = new Set<Listener>()
let observer: MutationObserver | null = null

/** 当前是否处于驾驶舱主界面模式。 */
export function getCockpitMode(): boolean {
  return cockpitMode
}

/** useSyncExternalStore 的 subscribe：返回退订函数。 */
export function subscribeCockpitMode(fn: Listener): () => void {
  listeners.add(fn)
  return () => { listeners.delete(fn) }
}

/** 翻转驾驶舱模式（头部按钮调用）。 */
export function toggleCockpitMode(): void {
  setCockpitMode(!cockpitMode)
}

/** 设定驾驶舱模式；变化时落 DOM 副作用并通知订阅者。 */
export function setCockpitMode(on: boolean): void {
  if (on === cockpitMode) return
  cockpitMode = on
  applyDom(on)
  for (const fn of listeners) fn()
}

function applyDom(on: boolean): void {
  const root = document.documentElement
  if (on) {
    root.setAttribute('data-re0-cockpit', 'on')
    markColumns()
    // dsh 重挂载 frame 会丢列标记 → 观察 DOM childList 变化补打（rAF 节流）。只观察
    // childList、不观察 attributes，故 markColumns 的 setAttribute 不会回环触发自己。
    if (!observer) {
      let scheduled = false
      observer = new MutationObserver(() => {
        if (scheduled || !cockpitMode) return
        scheduled = true
        requestAnimationFrame(() => {
          scheduled = false
          if (cockpitMode) markColumns()
        })
      })
      observer.observe(document.body, { childList: true, subtree: true })
    }
  } else {
    root.removeAttribute('data-re0-cockpit')
    observer?.disconnect()
    observer = null
  }
}

/**
 * 按稳定锚点给 frame 的三列打 data-re0-* 标记（幂等，可重复调用）。
 * AppFrame 的 DOM 子元素顺序（见 ui-layout/src/client/AppFrame.tsx）：
 *   sidebarCol | centerCol(main→conversation) | rightbarCol([data-rightbar-col])
 *   | overlayLayer([data-shell-overlay]) | 拖拽把手…
 * 故从 rightbarCol 往前两跳即得 centerCol / sidebarCol。
 */
export function markColumns(): void {
  const overlay = document.querySelector('[data-shell-overlay]')
  const frame = overlay?.parentElement ?? null
  if (!frame) return
  frame.setAttribute('data-re0-frame', '')
  const rightbar = frame.querySelector(':scope > [data-rightbar-col]')
  const center = rightbar?.previousElementSibling ?? null
  const sidebar = center?.previousElementSibling ?? null
  if (center) center.setAttribute('data-re0-center', '')
  if (sidebar) sidebar.setAttribute('data-re0-sidebar', '')
}
