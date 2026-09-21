/**
 * 驾驶舱主界面模式的注入 CSS（字符串，由 index.ts 塞进 <style data-plugin-css>）。
 *
 * 所有规则都限定在 `html[data-re0-cockpit="on"]` 下——默认（原版 dsh）零影响。几何用 CSS
 * 变量，live 调只改这里。依赖的 dsh 稳定锚点：`[data-shell-overlay]`（覆盖层，AppFrame 里
 * position:absolute;inset:0;z-index:20;pointer-events:none，子元素 auto）、以及 mode-store
 * 运行时打上的 `[data-re0-sidebar]`/`[data-re0-center]`（左栏列 / 中栏=聊天列）。
 */
export const COCKPIT_MODE_CSS = `
html[data-re0-cockpit="on"] {
  --re0-chat-w: 38vw;   /* 右上角聊天块宽 */
  --re0-chat-h: 60vh;   /* 右上角聊天块高（w×h≈1/4 屏） */
  --re0-chat-z: 21;     /* 必须 > overlayLayer 的 z-index:20，聊天才浮在驾驶舱上 */
}

/* 隐藏左侧栏列（会话/工作区/设置） */
html[data-re0-cockpit="on"] [data-re0-sidebar] {
  display: none !important;
}

/* 中栏（main→conversation，即 AI 对话）收到右上角一块，浮在驾驶舱之上 */
html[data-re0-cockpit="on"] [data-re0-center] {
  position: fixed !important;
  top: 0 !important;
  right: 0 !important;
  width: var(--re0-chat-w) !important;
  height: var(--re0-chat-h) !important;
  z-index: var(--re0-chat-z) !important;
  background: var(--dsw-alias-bg-base, #0d1117);
  border-left: 1px solid #30363d;
  border-bottom: 1px solid #30363d;
  border-bottom-left-radius: 10px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, .55);
  overflow: hidden;
}

/* 驾驶舱 iframe 填满覆盖层（overlayLayer 已 inset:0）；聊天块浮在其右上角之上。
   用后代选择器（非 >）：dsh slot 渲染器会在组件外套容器，iframe 不是直接子元素。
   （主尺寸由 CockpitOverlay 的内联包裹层保证，这里兼作兜底。） */
html[data-re0-cockpit="on"] [data-shell-overlay] .re0-cockpit-frame {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  border: 0;
  display: block;
  background: #0d1117;
}
`
