/**
 * 驾驶舱面板（会话头部动作）——点开一个 iframe 嵌 cockpit-server 的只读 web UI
 * （配置态结构树 / 事件流 / 告警）。数据不在这里取：iframe 直接加载 cockpit-server，
 * 它读 journal 文件（黑匣子），Hub 死后仍能看死前全貌。
 *
 * 自包含内联样式（不依赖 dsh 主题/外部 CSS）；暗色与 cockpit 一致。
 */
import { useState, type CSSProperties } from 'react'
import type { PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import { readBootstrap } from './bootstrap.ts'

type Props = PropsRuntime<'conversation.session.header.actions'>

const overlay: CSSProperties = {
  position: 'fixed', inset: 0, background: 'rgba(0,0,0,.55)', zIndex: 2147483000,
  display: 'flex', alignItems: 'center', justifyContent: 'center',
}
const panel: CSSProperties = {
  width: '92vw', height: '92vh', background: '#0d1117', border: '1px solid #30363d',
  borderRadius: 8, display: 'flex', flexDirection: 'column', overflow: 'hidden',
  boxShadow: '0 12px 48px rgba(0,0,0,.6)',
}
const head: CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 10, padding: '6px 12px',
  borderBottom: '1px solid #21262d', color: '#c9d1d9', fontSize: 13,
}
const frame: CSSProperties = { flex: 1, width: '100%', border: 'none', background: '#0d1117' }
const closeBtn: CSSProperties = {
  marginLeft: 'auto', background: '#21262d', color: '#c9d1d9',
  border: '1px solid #30363d', borderRadius: 5, cursor: 'pointer', padding: '2px 8px',
}

export function CockpitPanel(_props: Props) {
  const [open, setOpen] = useState(false)
  const url = readBootstrap()?.cockpitUrl
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title="Re0 机甲驾驶舱：配置态结构树 / 事件流 / 告警（只读，黑匣子）"
      >
        ◈ 驾驶舱
      </button>
      {open && (
        <div style={overlay} onClick={() => setOpen(false)}>
          <div style={panel} onClick={e => e.stopPropagation()}>
            <div style={head}>
              <b>Re0 驾驶舱（只读）</b>
              <span style={{ opacity: 0.6, fontSize: 11 }}>{url || '未注入 cockpitUrl'}</span>
              <button type="button" style={closeBtn} onClick={() => setOpen(false)}>✕ 关闭</button>
            </div>
            {url
              ? <iframe src={url} style={frame} title="Re0 cockpit" />
              : (
                <div style={{ padding: 16, color: '#8b949e', fontSize: 13, lineHeight: 1.7 }}>
                  host 未注入 cockpitUrl。确认：① cockpit-server 在跑
                  （<code>python -m src.cockpit.server</code>）；② 插件 <code>config.cockpitUrl</code> 已设；
                  ③ host 半已在 web 模式加载（<code>webserver/index-inject</code> 事件触发过）。
                </div>
              )}
          </div>
        </div>
      )}
    </>
  )
}
