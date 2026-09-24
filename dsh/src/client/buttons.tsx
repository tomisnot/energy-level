/**
 * 会话头部两个动作按钮：◈监控（开右栏 tab）与 → GUI（发 /re0-gui 交权人类）。
 *
 * 自包含内联样式（无外部 CSS 依赖，随 client bundle 打进合并包）。sessionId 是
 * header.actions 的 standardProp（dsh slot 契约），组件用它调 switchToGui。
 */
import { useState, type CSSProperties } from 'react'
import type { SessionId } from '@deepseek-ai/dsh-session/types'
import type { InjectFace, PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'

type HeaderProps = PropsRuntime<'conversation.session.header.actions'>

/** ◈监控 注入面：开右栏 + 打开我们的 tab（无状态动作）。 */
export interface CockpitInjected {
  openCockpit(): void
}
export type CockpitButtonProps = HeaderProps & InjectFace<CockpitInjected>

/** → GUI 注入面：发 /re0-gui（收 sessionId，返回错误串或 null）。 */
export interface SwitchInjected {
  switchToGui(sessionId: SessionId): Promise<string | null>
}
export type SwitchToGuiButtonProps = HeaderProps & InjectFace<SwitchInjected>

const BTN: CSSProperties = {
  padding: '2px 10px', borderRadius: 6, cursor: 'pointer',
  border: '1px solid rgba(128,128,128,0.35)', background: 'transparent',
  color: 'inherit', font: 'inherit', fontSize: '0.9em',
}

/** ◈监控：点击开右栏并打开机甲监控 tab。 */
export function CockpitButton({ openCockpit }: CockpitButtonProps) {
  return (
    <button type="button" style={BTN} title="打开机甲监控（右栏）"
      onClick={() => openCockpit()}>
      ◈ 监控
    </button>
  )
}

/** → GUI：点击请求切换到 GUI 模式（本 dsh 实例随交接退出）。 */
export function SwitchToGuiButton({ sessionId, switchToGui }: SwitchToGuiButtonProps) {
  const [msg, setMsg] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const onClick = async () => {
    if (busy) return
    setBusy(true)
    try {
      const err = await switchToGui(sessionId)
      setMsg(err ?? '已请求切换，dsh 即将退出…')
    } finally {
      setBusy(false)
    }
  }
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <button type="button" style={BTN} title="把控制权交还人类：切换到 GUI 模式（本 dsh 实例将退出）"
        disabled={busy} onClick={onClick}>
        → GUI
      </button>
      {msg ? <small style={{ opacity: 0.7 }}>{msg}</small> : null}
    </span>
  )
}
