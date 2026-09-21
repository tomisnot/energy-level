/**
 * `shell.overlay` 槽的 occupant：驾驶舱模式 ON 时渲染填满覆盖层的 cockpit iframe。
 *
 * shell.overlay 是 `{kind:'list', scope:'root'}`（见 ui-layout/src/client/index.ts 的 SlotMap），
 * 可与其它 overlay 并存。overlayLayer 自身 position:absolute;inset:0;z-index:20;pointer-events:none，
 * 且 `.overlayLayer > *{pointer-events:auto}`。数据不在这里取：iframe 直接加载 cockpit-server
 * （读 journal 的只读黑匣子）。OFF 时返回 null。
 *
 * ⚠ dsh 的 slot 渲染器会在注册的组件外套一层容器，iframe 并非 [data-shell-overlay] 的
 * 直接子元素——靠 CSS 直接子选择器选不中（曾缩成默认 300×150 贴左上）。故这里用内联样式
 * 让组件根容器对 overlayLayer（position:absolute）inset:0 铺满，iframe 再 100% 填充。
 */
import { useSyncExternalStore, type CSSProperties } from 'react'
import type { PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import { getCockpitMode, subscribeCockpitMode } from './mode-store.ts'
import { readBootstrap } from './bootstrap.ts'

type Props = PropsRuntime<'shell.overlay'>

const missing: CSSProperties = {
  position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
  justifyContent: 'center', padding: 24, textAlign: 'center',
  color: '#8b949e', fontSize: 13, lineHeight: 1.8, background: '#0d1117',
}

export function CockpitOverlay(_props: Props) {
  const on = useSyncExternalStore(subscribeCockpitMode, getCockpitMode)
  if (!on) return null
  const url = readBootstrap()?.cockpitUrl
  if (!url) {
    return (
      <div className="re0-cockpit-frame" style={missing}>
        host 未注入 cockpitUrl。确认：① cockpit-server 在跑
        （<code>python -m src.cockpit.server</code>）；② 插件 <code>config.cockpitUrl</code> 已设；
        ③ host 半已在 web 模式加载（<code>webserver/index-inject</code> 触发过）。
      </div>
    )
  }
  // 满尺寸绝对定位包裹层：对 overlayLayer（position:absolute）inset:0 铺满，不受 slot 外套容器影响
  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <iframe
        className="re0-cockpit-frame"
        src={url}
        title="Re0 机甲驾驶舱（只读）"
        style={{ width: '100%', height: '100%', border: 0, display: 'block', background: '#0d1117' }}
      />
    </div>
  )
}
