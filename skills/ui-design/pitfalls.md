# 踩坑清单（症状 → 原因 → 修法）

每条都在真实迭代中踩过。新坑请按此格式追加。

## PyQt5 / 离屏渲染

### 1. 中文全部渲染成方框 □□□
- **症状**：截图里所有 CJK 字符是空心方块，英文正常。
- **原因**：Qt offscreen 平台的字体库不含中文字体。
- **修法**：QApplication 创建后显式
  `QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')`，
  再 `app.setFont(QFont(families[0], 10))`。

### 2. import 主模块顺序颠倒 → 原生崩溃（exit 127，无 traceback）
- **症状**：脚本在 QApplication 之后 import 项目主模块时静默退出。
- **原因**：项目依赖的 native 库在 QApplication 已存在时的加载路径不同
  （项目相关，未必普遍——但遇到"无 traceback 的 127"先查这个）。
- **修法**：保持 `import 主模块` 在 `QApplication([])` 之前的顺序。

### 3. QMessageBox 挂死离屏脚本
- **症状**：脚本无输出卡死。
- **原因**：代码路径里的模态弹窗在 offscreen 下无人点确定。
- **修法**：脚本头部 `QMessageBox.information/warning = staticmethod(lambda …)`。

### 4. 面板被压扁（全屏模拟时）
- **症状**：截图里某面板只有 ~190px 宽，与真实全屏不符。
- **原因**：布局上限按 `QApplication.primaryScreen()` 算，offscreen
  虚拟屏只有 800×600。
- **修法**：截图脚本里按目标全屏分辨率手动重设该上限（不改生产代码——
  真实机器上屏幕尺寸=窗口尺寸没有问题）。

### 5. Qt 信号直连带参方法 → bool 被当业务参数
- **症状**：点按钮报 `AttributeError: 'bool' object has no attribute …`。
- **原因**：`clicked` 信号把 `checked` bool 作为第一个实参传入槽函数。
- **修法**：接线用 `lambda: self.method()`；方法内部对入参做类型防御。

### 6. 隐藏标签页里的画布尺寸为 0
- **症状**：对不可见 tab 里的 matplotlib/pyqtgraph 画布操作，布局塌缩
  （pyqtgraph 报 "axes sizes collapsed to zero"）。
- **修法**：先 `tabs.setCurrentIndex(目标页)` + processEvents，再操作画布。

## matplotlib

### 7. 图表四周大片留白
- **修法**：`Figure(figsize=…, constrained_layout=True)`（不要手动
  tight_layout 混用）。

### 8. 截图永远是旧画面（set_xlim/setTicks 不生效）
- **症状**：代码确认极限值已改、刻度数据已设，PNG 里还是旧帧。
- **原因**：`draw_idle()` 异步排程，`widget.grab()` 前缓冲未刷新。
- **修法**：grab 前 `canvas.draw()`（同步）。

### 9. 坐标轴刻度文字互相穿插
- **原因**：默认 ~10pt 字体下多行标签密度过高。
- **修法**：刻度字体显式降到 7pt + 密度收紧（视野 span/10 个）+
  `showGrid(alpha=0.18)` 提供标签↔数据对齐参照。

## pyqtgraph

### 10. import/构造 AxisItem 段错误（无 traceback 的 Segmentation fault）
- **修法**：任何 pg 对象构造前必须先建 QApplication。

### 11. `setAxisItem` 不存在
- **修法**：0.13.x 是 `plotItem.setAxisItems({'bottom': ax, 'left': ax})`。

### 12. 多行刻度标签整组消失
- **原因**：`style['hideOverlappingLabels']` 默认 True，多行文本
  bounding 判定互相重叠 → 全部隐藏；另外 `textFillLimits` 填充比
  不足时也会整组丢弃。
- **修法**：`ax.setStyle(hideOverlappingLabels=False,
  textFillLimits=[(0, 1.0)])`，密度自己控制。

### 13. 长刻度标签整组消失（隐形 contains 检查）
- **症状**：短标签正常、长标签全没——无报错。
- **原因**：generateDrawSpecs 里 `br = self.boundingRect();
  if not br.contains(textRect): continue`——文本 rect 超出轴宽被静默丢弃。
- **修法**：长名按固定宽度折行（如 22 字符/行），或加大
  `ax.setWidth(...)`。

### 14. ImageItem 方向/转置混乱
- **修法**：`ImageItem(axisOrder='row-major')` + `setImage(A.T)` 使
  `A[a, b]` 呈现为 a=纵轴（向上）、b=横轴；再用
  `img.setRect(QRectF(0, 0, dim, dim))` 固定数据坐标。

### 15. x 轴旋转刻度文字压进绘图区
- **原因**：pg 以"刻度矩形中心"定位文本，旋转后文本块上半截进入绘图区，
  而下方预留空间空着。
- **修法**：自定义 AxisItem 重写 `drawPicture`，锚点改为
  `translate(cx, rect.top() + text_width/2)` 后旋转——文本从轴线
  向下悬挂（自下而上阅读）。

## 交互/架构

### 16. 自适应刻度只该在"视野变化"时重算
- **修法**：matplotlib 挂 `ax.callbacks.connect('xlim_changed'/'ylim_changed')`；
  pyqtgraph 挂 `vb.sigXRangeChanged/sigYRangeChanged`。重算逻辑纯函数化，
  视野 span 决定密度与详细程度（span≤30 → 逐维全标注）。

### 17. 全局常量覆写不要改模块属性
- **症状**：改 `params.BW_OPT` 后引擎行为不变。
- **原因**：消费方 `from .params import BW_OPT` 是按值绑定进自己命名空间。
- **修法**：覆写走函数 kwargs（运行级生效、不污染全局），默认值仍从
  params.py 实时读取展示。
