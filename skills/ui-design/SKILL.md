# SKILL：AI 自主 UI 设计反馈工作流（截图自查闭环）

> 让没有人类反馈的 AI 独立完成 GUI 迭代：**改 UI 代码 → 离屏渲染真实界面 → 截图 →
> 视觉自查 → 修正 → 回归**。本 skill 来自一个 PyQt5 科学计算软件的真实迭代过程
> （20+ 轮反馈闭环全部由 AI 自主完成），可直接移植到任何 PyQt5 项目。

---

## 什么时候用

- 需要新增/修改 GUI 界面，但没有人实时给你反馈；
- 用户只给方向性意见（"做个 XX 面板"），需要自主验证视觉效果；
- 排查"界面为什么显示异常"类问题。

前置条件：项目是 PyQt5（PySide 同理）；机器是 Windows（字体路径相关）；有 Python。

## 核心工作流（五步闭环）

```
① 改 UI 代码
② 写/跑截图脚本（离屏渲染真实 MainWindow → PNG）
③ 读 PNG（视觉模态）→ 对照自查清单找问题
④ 修复 → 重截 → 直到清单全过
⑤ 跑项目原有 UI 回归测试（确保没破坏旧功能）
```

**关键纪律**：
- 涉及可见界面变化时过一遍相关排版自查（见 checklist.md）——只验证"功能对"
  不验证"排版好"会被用户一眼看穿；
- 每轮结束跑一遍项目原有 UI 回归测试；
- 把每轮发现的问题和修法记进 pitfalls.md——它们会反复出现。

## 文件说明

| 文件 | 用途 |
|---|---|
| `shot_workflow_template.py` | 离屏截图脚本模板，改三个函数即可适配任何项目 |
| `pyqtgraph_adaptive_heatmap.py` | pyqtgraph 自适应刻度热图组件（缩放随动标注 + 竖排轴），含可运行 demo |
| `demo_effect_zoom.png` | 上述组件局部放大的实际渲染效果（75 维矩阵缩到 16 维窗口，逐维全标注） |
| `pitfalls.md` | 踩坑清单：症状 → 原因 → 修法（Qt / matplotlib / pyqtgraph，共 17 条） |
| `checklist.md` | 截图自查清单（排版关 + 功能关 + 回归关） |
| `ui_patterns.md` | 已验证的 UI 设计模式（检视器分页/大卡片/旋钮语义位置/自适应刻度等 9 条） |

## 快速上手（三步接入新项目）

1. 复制 `shot_workflow_template.py` 到项目（如 `checks/ui_shot.py`），
   填三个函数：`build_window()`（怎么构造主窗口）、`load_scenario(win)`
   （怎么填充真实数据）、`capture_list(win)`（要截哪些页/对话框）；
2. `python checks/ui_shot.py` → 生成 PNG；
3. AI 用视觉读 PNG，按 `checklist.md` 自查，改代码，重跑。

## 截图的三个铁律（违反必踩坑）

1. **CJK 字体**：offscreen 平台默认字体库无中文，必须显式
   `QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')`，
   否则所有中文渲染成方框（见 pitfalls #1）；
2. **导入顺序**：本项目实测 import 主模块必须在 QApplication 之前，
   顺序颠倒会原生崩溃（exit 127）——每个项目要先测出安全顺序；
3. **同步绘制**：matplotlib `draw_idle()` 是异步的，`widget.grab()` 前
   必须 `canvas.draw()` 同步重绘，否则截到旧帧（见 pitfalls #8）。

---

## 本项目（Energy Level）的具体对接 —— **先看这里，别重写脚本**

`ui_design/` 里已经有成套工具，直接调用即可：

| 脚本 | 用途 |
|---|---|
| `ui_shot.py` | 主窗口两页（静态/动态）离屏快照 → PNG |
| `ui_shot_ab.py` | 截图平台 A/B 对比（offscreen vs 真实 windows 平台） |
| `ui_shot_god.py` | 上帝模式（诊断检视器）页快照 |
| `shot_hamiltonian.py` | 哈密顿量显示页：改 → 截 → 自查 → 重跑 |
| `layout_audit.py` | 布局体检：机器可读的控件几何/文本 dump（**不靠看图**） |
| `font_audit.py` / `font_glyph_probe.py` / `font_render_probe.py` / `font_fallback_audit.py` | 缺字/回退体探针（offscreen 与真实平台各一套） |
| `crop_zoom.py` | 截图局部裁剪放大（供视觉模态逐像素看） |
| `debug_geom.py` | 几何断言（不依赖看图） |

**更快的一条路**：软件实例在跑时，AI 可直接用 MCP 工具 `screenshot()` 拿**当前主窗口**
PNG（不必构造窗口/填数据）——用于"真人此刻看到什么"；需要**指定页面/多页对比**时再用上表脚本。

**本项目实测的"安全顺序"**（对应上面铁律 2 的具体值）：
`import 能级 as main_mod` **必须在 `QApplication` 之前**，颠倒会原生崩溃（exit 127）。
截图脚本一律 `QT_QPA_PLATFORM=offscreen` + `EL_NO_AI_SERVER=1`。

**CJK 字体**：本项目用 `C:/Windows/Fonts/msyh.ttc`（微软雅黑）。matplotlib 另需
`plt.rcParams['font.sans-serif']` 显式指定并设 `axes.unicode_minus = False`
（否则负号与上标 σ⁺ 之类会成方框——`σ⁺/σ⁻` 用 `σ+/σ-` 写更稳）。

**回归方式**：UI 改动运行受影响场景，例如
`python run_checks.py --only gui_`；完整验证范围见 `checks/README.md`；
涉及界面行为的**用户裁决项**（例如"那一行不得有某某按钮"）已有自检钉住——别只靠看图。

**出图（figure）约定**：战役图统一放 `ui_design/screenshots/`；多面板并排时按物理对照组
从左到右排、标题写清 `(ω, I, 偏振)`，并在图注里给出判据阈值（例：`C_dual ≥ 0.9`）。
