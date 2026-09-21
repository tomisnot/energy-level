/**
 * 会话头部动作：驾驶舱主界面的**切换开关**（不是弹窗）。
 *
 * 点一下把主界面切成驾驶舱（驾驶舱占主区、AI 对话收到右上角约 1/4、隐藏左侧栏）；再点切回
 * 原版 dsh。状态放在模块级 mode-store，与 shell.overlay 里的 CockpitOverlay 共享（两者 scope
 * 不同，只能模块级桥接）。自包含、无外部 CSS 依赖。
 */
import { useSyncExternalStore } from 'react'
import type { PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import { getCockpitMode, subscribeCockpitMode, toggleCockpitMode } from './mode-store.ts'

type Props = PropsRuntime<'conversation.session.header.actions'>

export function CockpitPanel(_props: Props) {
  const on = useSyncExternalStore(subscribeCockpitMode, getCockpitMode)
  return (
    <button
      type="button"
      onClick={() => toggleCockpitMode()}
      data-re0-cockpit-toggle={on ? 'on' : 'off'}
      title={on
        ? '退出驾驶舱主界面，回到原版 dsh 布局'
        : '把主界面切成驾驶舱：驾驶舱占主区、AI 对话收到右上角、隐藏左侧栏'}
    >
      {on ? '◈ 退出驾驶舱' : '◈ 驾驶舱'}
    </button>
  )
}
