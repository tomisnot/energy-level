/**
 * 驾驶舱主界面模式的注入 CSS（字符串，由 index.ts 塞进 <style data-plugin-css>）。
 *
 * 设计：驾驶舱 = dsh **右侧栏**。iframe 由 index.ts 挂进右栏列 `[data-rightbar-col]`，
 * 这里隐藏右栏原有 dock 内容、让 iframe 填满右栏列。对话回中栏全高（不 dock），
 * 仅精简其 chrome（隐藏 crumbs/tabs/utilities/corner/底部统计 + 头部压到按钮高）。
 * 全部限定在 `html[data-re0-cockpit="on"]` 下，默认（原版 dsh）零影响。
 */
export const COCKPIT_MODE_CSS = `
/* 右栏=驾驶舱：隐藏右栏原有 dock 内容，只留 cockpit iframe 填满右栏列 */
html[data-re0-cockpit="on"] [data-rightbar-col] > *:not(.re0-cockpit-frame) {
  display: none !important;
}
html[data-re0-cockpit="on"] [data-rightbar-col] > .re0-cockpit-frame {
  width: 100%;
  height: 100%;
  border: 0;
  display: block;
  background: #0d1117;
}

/* 对话 chrome 精简：隐藏 crumbs(标题/模式/费用)/tabs(对话/轨迹)/utilities/corner/底部统计 */
html[data-re0-cockpit="on"] [data-re0-center] header [role="tablist"] { display: none !important; }
html[data-re0-cockpit="on"] [data-re0-center] [data-conversation-header-corner] { display: none !important; }
html[data-re0-cockpit="on"] [data-re0-center] header nav { display: none !important; }
html[data-re0-cockpit="on"] [data-re0-center] header > :first-child > :nth-child(2) { display: none !important; }
html[data-re0-cockpit="on"] [data-re0-center] [data-composer-stats] { display: none !important; }
/* 头部压到“放得下按钮”的高度（隐藏两行 chrome 后去掉空白） */
html[data-re0-cockpit="on"] [data-re0-center] header {
  min-height: 0 !important;
  height: auto !important;
  padding: 2px 8px !important;
}
html[data-re0-cockpit="on"] [data-re0-center] header > :first-child {
  min-height: 0 !important;
  height: auto !important;
  padding: 0 !important;
}
`
