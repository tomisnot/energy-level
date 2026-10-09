/**
 * 会话头部动作按钮：◈监控（开右栏 tab）。
 *
 * 自包含内联样式（无外部 CSS 依赖，随 client bundle 打进合并包）。
 *
 * ⚠ **曾经还有「→ GUI」按钮**（发 `/re0-gui` 交权人类、看门人接手进程交接）。
 * **三层架构（2026-10-08）删了整条链**：切写权走**控制端点**、**不动进程**
 * ⇒ 那个按钮没有存在理由（留着就是"按了没人接"）。
 */
import type { CSSProperties } from 'react'
import type { InjectFace, PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import { EL_HEADER_LABEL } from '../el-params.ts'

type HeaderProps = PropsRuntime<'conversation.session.header.actions'>

/** ◈监控 注入面：开右栏 + 打开我们的 tab（无状态动作）。 */
export interface CockpitInjected {
  openCockpit(): void
}
export type CockpitButtonProps = HeaderProps & InjectFace<CockpitInjected>

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
      {EL_HEADER_LABEL.cockpit}
    </button>
  )
}
