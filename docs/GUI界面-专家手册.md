# GUI 界面模块 · 专家手册

> ⚠ **历史件横幅（2026-09-25 加，主代理指令）**：本文记录的是**已退役的 `mecha-sdk` 0.1.0**
> 时代（`D:\学\BaiduSyncdisk\编程\code\python\mecha-sdk`）。EL 现运行的框架是 **`mecha` 0.2.0**
> （editable 装于 `D:\code-nosync\mecha`），**模块集完全不同**：0.1.0 的 `ops` / `session` /
> `bus` / `registry` / `codeact` / `server` / `cockpit` / `hub_bridge` / `hub_client` /
> `archives` / `interlock` / `autonomy` / `gc` / `app_bridge` / `subscribe` / `hooks` 等在
> 0.2.0 **都不存在**（改由 `gate` / `history` / `authority` / `surface` / `tools` /
> `assembly` / `data_layout` / `providers.*` 承担）。
> ⇒ 本文所有 `mecha-sdk/...py:行号`、`src/adapter/*`、`cockpit *` 的路径与行号**均已失效**
> （GUI 的 v1 侧栏监控、Hub 客户端模式等描述也已随 v1 退役），只作历史记录，
> **请勿据以定位代码**（主代理要求：不机械替换，只加本横幅）。
> 当前真值源：本仓 `docs/capabilities.md`（生成物）+ mecha 仓 `docs/v2/`。

> **模块边界**：`app_entry.py`（仓根，546 行）+ `src/ui/*`（15 个 .py，8588 行，含 `__init__.py`）
> + `能级.py`（转发入口）+ GUI 判据 6 个 + `dsh/src/client/*`（驾驶舱，仅在与界面体验相关时）。
> （`main.py` 与打包路线已于 2026-09-25 删除；详见文首横幅。）
> **不属本模块**：物理引擎（`src/core`）、AI 机甲框架（mecha-sdk）、sweep 的 runner 逻辑。
>
> **实测基线（2026-09-22 本机复核）**：`run_checks.py --only gui_` → **scripts=6 fail=0 seconds=31**
> （`gui_app_entry` 1 PASS / `gui_bind` 5 / `gui_inspector` 3 / `gui_run_flow` 3 / `gui_run_worker` 2 / `sweep_gui` 6）。
> 主代理全量基线：`scripts=46 fail=0 seconds=589`。
> `cleanliness_guard_selfcheck` 单跑 → 1 PASS / 0 FAIL。
>
> **口径**：本文所有结论带 `文件:行号`，并区分「文档声称」与「代码实际」。mecha 侧文件在仓库外，
> 路径写作 `mecha-sdk/mecha/xxx.py`（实装位置 `D:\学\BaiduSyncdisk\编程\code\python\mecha-sdk\mecha\`）。

---

## 0. 速查

1. **启动**：`python 能级.py` → `能级.py:12` `from app_entry import MainWindow, main` → `app_entry.main()`（`app_entry.py:530`）
   → **模态** `_choose_mode()`（`app_entry.py:507`）选 AI/GUI → `run_gui(app)`（`app_entry.py:403`）= `MainWindow()` + `showMaximized()`。
   （直接跑：`python app_entry.py`。原「打包：`main.py:5`」一路已于 2026-09-25 随打包路线删除。）
2. **`MainWindow.__init__`**（`app_entry.py:72`）：先定 transport（`_hub_connected` / `HubBridge` vs `AppBridge`，`:84-96`）→ `cache`/`binder`（`:97-98`）→ `_build_ui()`（`:185`）。
3. **页面**：中央 `canvas_tabs` 两页 = **静态**（`analysis_panel.canvas` = `EnergyPlotPanel`）/ **动态**（`DynamicPanel`，`app_entry.py:238-247`）；
   右侧 `panel_scroll` = 能级组面板 + 场面板 + `page_stack`（静态/动态各自的参数区，`:235-236,259-264`）。
4. **底部**：状态栏常驻进度条+阶段标签（`:206-219`）；`sweep_stack` 在 `top` 之后、底栏正上方（`:255-257`）。
5. **两个"模式"是两回事**：**视图模式** 极简/上帝 = 菜单栏右上角互斥钮（`_build_mode_switch`，`:281`）；
   **写权模式** human/AI = `app_bridge.authority`，**GUI 今天没有任何入口**（见 §5）。
6. **写状态的唯一 choke point** = `app_bridge.write/write_many` → `session.set/set_many`（actor+reason 必填）。
7. **跑演化**：`DynamicPanel._run()`（`ui_dynamic.py:795`）→ 主线程 `ops.prepare` → `RunWorker`（QThread）`ops.execute`
   → artifact → `_on_evolve_done` 取回绘图（`:870-882,1412`）。Hub 模式改走 `HubRunWorker`（`:858-868`）。
8. **改界面先读 §7 布局红线 + §10 安全工序**；`--only gui_` 已经包含 `sweep_gui`（子串匹配，6 个）。

---

## 1. 启动与装配

### 1.1 启动链路（逐跳）

| 跳 | 位置 | 说明 |
|---|---|---|
| 1 | `能级.py:12` | 只转发 `app_entry.main`；`能级.py` 额外再导出 `MainWindow`（供 `import 能级 as main_mod` 的旧脚本）。**`main.py` 与打包路线已于 2026-09-25 删除** |
| 2 | `app_entry.py` 顶部 | 无控制台启动守卫：`sys.stdout/stderr is None` → 补 devnull |
| 3 | `app_entry.py:530-536` | `QApplication` + `ui_theme.QSS`（`ui_theme.py:60`，字体 10pt Microsoft YaHei） |
| 4 | `app_entry.py:537` | `_choose_mode()` **模态** QMessageBox：AI 模式 / GUI 模式 / 取消 |
| 5a | `app_entry.py:539` → `403` | GUI：`MainWindow()` + `showMaximized()` + `app.exec_()` |
| 5b | `app_entry.py:541` → `410` | AI：后台 spawn `-m src.adapter.serve`（Hub :8765）+ `-m cockpit.server`（:8766），前台跑 `dsh web --patch ./dsh/cordis.source.patch.yml`；等 `.mcp-port` 最多 ~20s；缺 `dsh/node_modules` 自动 `npm install`（`:470-477`） |
| 6 | `app_entry.py:84-96` | `MainWindow.__init__` 定 transport（见 §1.3） |
| 7 | `app_entry.py:97-98` | `cache = app_bridge.cache`；`binder = _mecha_bind.Binder(app_bridge)` |
| 8 | `app_entry.py:113,185` | `_build_ui()` 装配全部面板 |
| 9 | `app_entry.py:274` | `QTimer.singleShot(0, self._apply_fullscreen_layout)` → 布局定型（DPI/面板宽度/最小高度，`:386-400`） |

**注意**：`run_ai()` 是**启动器**（起服务 + 前台 dsh），不是"GUI 只读模式"。`app_entry.py:411-416` 的 docstring
说驾驶舱是 dsh 网页头部 iframe —— **已过时**，现行是右栏 tab（`dsh/src/client/index.ts:8`「无 iframe、单一实现」）。

### 1.2 `_build_ui` 装配顺序（改装配必须照这个顺序读）

`app_entry.py:185-274`：
1. `main_widget` = centralWidget，`root = QVBoxLayout`，`top = QHBoxLayout`（`:187-191`）。
2. 三个面板先建：`EnergyGroupPanel`（`ui_group.py:15`）、`LevelAnalysisPanel`（`ui_spectrum.py:13`）、`DynamicPanel`（`ui_dynamic.py:44`）；
   再连 `canvas.peakPicked`/`peakFreqsChanged`/`groups_changed`（`:193-198`）。
3. `FieldPanel`（`fields.py:23`）+ `data_changed → _on_light_changed`（`:200-201`）。
4. **状态栏常驻进度**：`_run_label`(QLabel '空闲') + `_run_pbar`(QProgressBar, 0..1000, 固定宽 180)（`:206-219`），
   并把同一对控件注入 `dynamic._stage_label` / `dynamic._pbar`（`:218-219`）——**人与 AI 共用这两个控件**。
5. `LevelGroupStore` / `FieldState` 建起并 `bind_store`（`:221-226`）。
6. `control_widget`（`SizePolicy.Ignored × Preferred`，spacing 2）= 能级组 + 场 + `page_stack`（`:228-236`）。
7. `canvas_tabs` + `modules` 表（每项 7 元组：name/tab_widget/panel_widget/on_show/on_light/use_field/field_lights，`:239-247`）。
8. `SweepStack` 加进 `root`（`top` 之后 ⇒ 底栏正上方，`:255-257`）。
9. `panel_scroll`（`QScrollArea`，`widgetResizable`，横向滚动条关）包 `control_widget`，`top.addWidget(canvas_tabs, 3)` / `top.addWidget(panel_scroll, 2)`（`:247,259-264`）。
10. `_on_canvas_tab_changed(0)` → 菜单（文件→保存/恢复）→ `_build_mode_switch(menubar)` → `QTimer.singleShot(0, …)`（`:266-274`）。

### 1.3 GUI 是"Hub 客户端"还是"自托管内核"

**两条路径由 `.mcp-port` + 环境变量决定，代码在 `app_entry.py:84-96`**：

```
RE0_GUI_INPROC 置位 → 强制 in-process（判据/离线；run_checks.py:24 统一设 1）
否则 discover_hub() 读 ./.mcp-port（mecha-sdk/mecha/hub_client.py:125-133）
   ├─ 读到 → HubBridge(url)（mecha-sdk/mecha/hub_bridge.py:26）→ _hub_connected=True
   └─ 读到但 initialize() 抛异常 → except → AppBridge(engine=EngineAdapter(), mode='human')  ← graceful 回落
   读不到 → AppBridge(...)
```

- **in-process**（默认人工模式）：GUI 进程内自托管 Bus/Authority/Session/Ops/Jobs（`mecha-sdk/mecha/app_bridge.py:39-47`），
  `cache` 是本地引擎 cache，`ops.prepare/execute` 都在本地（`ops.py:152,178`）。
- **Hub 客户端**：`HubBridge` 把方法面映射到 MCP 工具（`hub_bridge.py:5-14`）；`cache`/`fields`/`session` 全返回 `None`，
  `ops` 返回 `self`（`hub_bridge.py:36-53`）。GUI 用 `getattr(self.main.app_bridge, 'client', None) is not None` 判断分支（`ui_dynamic.py:858`）。
- **实测现状（本机）**：`.mcp-port` 存在（内容 `8765`）但 8765 无监听 → `MainWindow` 实测 `_hub_connected=False`、`bridge=AppBridge`
  （offscreen 探针）。**陈旧端口文件只造成一次快速连接失败 + 回落**，不影响可用性。
- `_hub_connected` 是**死标志**：只被赋值（`app_entry.py:86,92`），全库无任何 UI 读它（grep 仅 2 处）。

**Hub 路径的风险（代码读出的结论，本轮无活 Hub，未做端到端实测）**：
1. **总线订阅形状不匹配 → Hub 模式实时刷新彻底失效**：`Binder.__init__` 订阅 `bridge.subscribe(self._on_event)`（`_mecha_bind.py:63`），
   而 `HubBridge.subscribe` 回调的是 **list（变更键名列表）**（`hub_bridge.py:89-109` 的 `cb(changed)`），
   `_on_event` 期待 **dict**（`_mecha_bind.py:76` `ev.get('kind')`）。实测：`Binder._on_event(['field.B_gauss'])`
   → `AttributeError: 'list' object has no attribute 'get'`。该异常发生在 `subscribe` 的守护线程里，
   被 `bus` 层保护不到（`hub_bridge.py:93-106` 的 try 只包 `effective()`）→ **线程静默死掉**。
2. **GUI 写入被记成 AI**：Hub 侧 MCP 工具的 actor 硬编码 `'ai'`（`mecha-sdk/mecha/server.py:94,148`），
   `write_many` 走 `mecha_script` → codeact 门面把 actor 固定 `'codeact'`（`mecha-sdk/mecha/codeact.py:106`）。
   `HubBridge.write/write_many` 收下 `actor='gui'` 参数但**丢弃不用**（`hub_bridge.py:112-117`）。
   ⇒ 人在 Hub 模式的写入在 journal 里是 AI 侧动作（M1 归因错），且 `Binder` 会把**人自己**的改动当成 AI 改动去闪紫框。
3. **写失败静默**：Hub 把 `AuthorityError` 变成 `{'ok': False, 'error': …}` 回程（`server.py:123-132`），
   HTTP 仍是 200；GUI 侧 `_mecha_bind.py:71` 与 `ui_dynamic.py:936` 都**不检查返回值** ⇒ human 模式的 Hub 上，
   人的写入静默无效（违反 D5「报错不静默」）。反过来说：Hub 模式**不会**因写权被拒而崩 Qt 槽。
4. `HubBridge.switch_mode` 存在（`hub_bridge.py:136-154`，走 `.human-token` + 固定 :8767），但 **GUI 从不调用它**（见 §5）。

### 1.4 死开关与陈旧注释（不要被它们误导）

| 位置 | 状态 |
|---|---|
| `EL_NO_AI_SERVER` | **已失效**（src 内无任何读取点；`serve.py:85` 用的是 `RE0_MECHA_NO_SERVER`）。判据仍在设它（`gui_app_entry_selfcheck.py:18` 等），无害但无作用 |
| `AppBridge.mcp_server()` | GUI **不调用**（全库无 `mcp_server` 命中）。`app_bridge.py:10` 与 `app_entry.py` 旧注释说的"进程内 MCP 后台线程"已被 T4 修正（Hub 独立进程）取代 |
| `MainWindow.closeEvent` | 已无服务可停（`app_entry.py:120-121`）；旧版这里 `stop_server()` |
| `_show_progress()` | 空实现（`ui_dynamic.py:1205-1210`），但 `_dlg_timer`(400ms) 仍在启动（`:821-824`）、`_preflight_cost` 仍判 `_dlg`（`:1317-1320`）——全是死分支 |
| `_ai_initiated` | **从未被赋值**（仅 `ui_dynamic.py:1263` docstring 与 `:1307` 读取）。⇒ 「AI 发起不弹模态」的保护**没接线**；若真让 AI 走 `_run()`，成本报告 `QMessageBox`（`:1322-1333`）会弹出（违反"AI 路径不弹模态"） |
| `app_entry._on_engine_param_edited` | 定义了但**无任何连接点**（`:332`） |
| `solver_inspector.param_edited` 信号 | 声明了（`solver_inspector.py:262`）但**从不 emit、从不 connect** |
| `Binder.push_all()` / `register_external()` | GUI **从不调用**（实测 `MainWindow.binder._ext == 0`）。非标参数改由运行前 `write_many` 一次性同步（`ui_dynamic.py:895-936`） |
| `sweep_panel` docstring 提到的 `EL_NO_AI_SERVER` | 同上，失效 |
| `docs/手册-界面·原理·开发.md:66-77` 的布局图 | **左右反了**：代码是画布左、控制面板右（`app_entry.py:247,264`）；该图还写"存档 v3 登记册即存档"（现行是 `mecha-session-v1`） |

---

## 2. 页面与控件地图

### 2.1 主窗口骨架

```
QMainWindow(MainWindow, 1400×800, 标题 'Rb-87 能级计算')
├─ menuBar: [文件: 保存状态 / 恢复状态]        cornerWidget(TopRight): [极简模式|上帝模式]  ← app_entry.py:269-304
├─ statusBar: showMessage 出口 + addPermanentWidget(_run_label '空闲', _run_pbar 0..1000 宽180)  :79,115-118,215-216
└─ centralWidget
   └─ QVBoxLayout root
      ├─ QHBoxLayout top  (stretch 1)
      │  ├─ canvas_tabs (QTabWidget, stretch 3)  ← 静态 / 动态   :238-247
      │  └─ panel_scroll (QScrollArea, stretch 2) ← 右侧控制面板 :259-264
      │     └─ control_widget (SizePolicy.Ignored × Preferred)
      │        ├─ EnergyGroupPanel      :232
      │        ├─ FieldPanel            :233
      │        └─ page_stack (QStackedWidget, stretch 1)  :235-236
      │           ├─[0] LevelAnalysisPanel（静态页参数区）
      │           └─[1] DynamicPanel.dynamic_widget（动态页参数区）
      └─ SweepStack（并行扫描栏；无扫描时不占高度）  :255-257
```

`modules` 表驱动切页（`app_entry.py:239-242`）：切到"静态"时 `field` 可见 + 光场灯可见；切到"动态"时 `field` 可见但**光场灯隐藏**
（`field_lights=False`，`:346-351`）。

### 2.2 各面板

| 面板 | 文件:类 | 主要内容（控件级） |
|---|---|---|
| 能级组 | `ui_group.py:15` `EnergyGroupPanel` | n/l/j spin + 标签预览 + 「添加」（`:29-46`）；`group_list`（右键菜单=删除/「子能级…」裁剪，`:52-55,218`）；`level_table`（能级组/mJ/E(MHz)，`:57-68`）。**注意**：`ui_group.py:47-49,73-99` 明确记录了「逐子能级裁剪没有工具条按钮」「基组预设+维数提示整行已删」两条用户裁决——**别再把它们加回来** |
| 场 | `fields.py:23` `FieldPanel` | B/E（**经 MB 绑到 `field.B_gauss`/`field.E_vpcm`**，`:50-66`）；「窗口」`window_spin` 与「光子块」`n_photon_spin`（`:72-94`）；「＋添加光」；光场表 5 列（`#/频率GHz/光强mW·cm⁻²/偏振/作用范围`，`:111-120`）；关注跃迁 combo；初态 n/l/j/m；右键「从跃迁对导出频率」「扫描光强实现目标频移」（`FreqFromTransitionDialog:747` / `IntensityScanDialog:802`） |
| 静态 | `ui_spectrum.py:13` `LevelAnalysisPanel` | 画布 `EnergyPlotPanel`（`ui_single.py:27`）；顶部一行：能级图模式 combo（单一/off·on 双图）、`y↓/↑` 组选择、参考 GHz、谱 x 范围（`:29-48`）；`peak_pair_label`（`:51-52`）；`spec_off_table`/`spec_on_table`（4 列，`:54-55,100-119`）；耦合雷达 `CouplingRadarPlot`（`ui_radar.py:25`，`:57-58`） |
| 动态 | `ui_dynamic.py:44` `DynamicPanel` | 上画布=光强包络（A/B 双子图，`:95-111`）、下画布=布居/读出；控制区见下 |
| 波形 | `ui_waves.py:61` `WavePanel` | 三件套：主列表 + `[▸辅助]` 折叠区（g/e 态 + 「定光强」`setI_b:794` + 「定频率」`setF_b:805`，文件头 `:13-14` 说明"辅助区不参与结算"）；三个独立实例 `wave_panel` / `_A` / `_B`（`ui_dynamic.py:199-204`） |
| 扫描栏 | `sweep_panel.py:53` `SweepStack` | 表头（常显）+ 逐点行（可收放）；`MAX_ROWS=16`、`BODY_MAX_H=190`、`IDLE_HIDE_S=8`、`POLL_MS=1000`（`:39-42`）；只读 `src.sweep.registry.status()`，取消只写 cancel 文件（`:22-25`） |
| 检视器 | `solver_inspector.py:252` `SolverInspector` | 见 §6 |
| 哈密顿量 | `hamiltonian_viewer.py:77` `HamiltonianViewer` | 检视器"哈密顿量"页内嵌的 pyqtgraph 场景层（工具条 + 热图 + `_ColorBar:1477`） |

### 2.3 动态页控件清单（`ui_dynamic.py:114-397`，按屏幕从上到下）

1. `_vdd_box`：R / θ / φ（**MB 绑** `dynamic.pair.R_um/theta_deg/phi_deg`，`:128-147`），默认隐藏。
2. 单原子 / 双原子 单选行（`_mode_group`，`:150-162`）；V_dd 框紧跟"双原子"（t25 同排）。
3. `_sambe_adv_box`：`frozen_h(ns)`（**MB 绑** `solve.FROZEN_H_NS`，`:171-178`）+ `per_freq_M` QLineEdit（**刻意不绑**，因 session 键是 `int_or_list`，运行前解析后写，`:180-187`）。
4. 波形三面板 + A/B `QTabWidget`（`:199-226`），tabbar 装 eventFilter 支持拖放复制。
5. 初态行：`psi_combo`（单）/ `_psi_box`(A/B) + 「自发辐射」`decay_mode_cb`（自动/禁用/自定义，`:244-270`）。
6. 时间/方法行：`t_max`、`npts`（**MB 绑**）、`method_cb`（**MB 绑** `dynamic.method`，档位 `sambe/hard/mixed`，`:295-298`）、`mode_badge`（`:305-311`）。
7. 自定义衰变折叠区（`:316-347`，仅"自定义"档显示）。
8. 显示选项行：`display_mode_cb`（态布居/荧光读出）+ `topn_sb` + `readout_cb`（宽 300，末尾哨兵"复杂配置…"）+ **运行按钮**（运行中变"取消计算"，`:350-394`）。

**状态真值说明**：MB 绑定的 9 个键实时写 session；**非标量**（`groups.list`/`dynamic.mode`/`waves_*`/`psi_*`/`decays`/`pair.*`/`solve.PER_FREQ_M`）
在**运行前**由 `_sync_run_config()` 一次性 `write_many`（`:895-936`）。存档 = `app_bridge.save`（session 快照），
恢复 = `app_bridge.load` + `MainWindow.refresh()`（`app_entry.py:363-384`）。**面板不再持有 serialize/restore 平行清单**（`:714-716`）。

---

## 3. `_mecha_bind` 绑定层（287 行，取代 `pm_setup` + 外部 `param_manager`）

| 件 | 位置 | 行为 |
|---|---|---|
| `ParamSpec` | `_mecha_bind.py:27` | 近 drop-in 的元数据壳（kind/default/unit/desc/aliases/options/group/lo/hi/log/tier）；**GUI 侧只用于提示与类型** |
| `Binder.__init__` | `:56-63` | `bridge.subscribe(self._on_event)`；`_binds=[(widget,key,setter)]`、`_ext={}`、`_updating` 防回环、`_flash_timers` |
| 写 | `:66-72` | `_updating` 中直接返回；`can_write('gui')` 为假**静默 return**；否则 `bridge.write(..., actor='gui')` |
| 订阅 | `:75-94` | 只看 `kind in ('set','undo')`；`actor` 以 `gui` 开头 → 跳过（避免自己闪自己）；AI 侧（`ai/codeact/job/mcp`，`:23`）→ `_refresh_key(flash=True)`；`_ext` 命中则 `apply_fn(bridge.read(key))` |
| 归因高亮 | `:109-127` | 紫框 `border: 2px solid #a855f7`（`:24`），`QTimer` 2s 后还原；原 QSS 存 `_mb_base_qss` 属性（**不回流布局**） |
| 只读门控 | `:130-148` | spin/lineedit → `setReadOnly`；combo/checkbox → `setEnabled(False)` |
| 工厂 | `:151-209` | `make_spinbox/combobox/lineedit/checkbox` + 模块级 `spinbox/intbox/combobox/lineedit/checkbox`（`:266-287`），签名对齐旧 `pm.*` |
| 非标登记 | `:212-250` | `register_external` / `_seed`（用 `actor='session'` 种子，不过门）/ `push_all` —— **GUI 未使用**（`_ext` 实测为空） |

**实际绑定面（全库 grep 实测）**：`field.B_gauss`、`field.E_vpcm`（`fields.py:50,59`）；
`dynamic.pair.R_um/theta_deg/phi_deg`、`solve.FROZEN_H_NS`、`dynamic.t_max`、`dynamic.npts`、`dynamic.method`（`ui_dynamic.py:128,135,142,171,275,282,295`）。
⇒ **共 9 个**（实测 `len(binder._binds)==9`）。
**未绑定的 GUI 状态**（能级组、波形、初态、衰变、psi、per_freq_M、读出权重、能级图显示选项）**不参与实时刷新与归因高亮**。

---

## 4. 运行流（QThread / HubRunWorker / 进度 / 取消）

### 4.1 人机同一条 `_run()`

- 唯一入口 `DynamicPanel._run()`（`ui_dynamic.py:795`），按钮 `_on_run_clicked`（`:1186`）运行中即取消（`_cancel_run:1212`）。
- 进度协议 `progress_arr = [阶段内进度, 取消标志, 阶段]`（`:810`）：`stage 0 准备 / 1 积分 / 2 收尾`（`:1221-1253` 映射到 0-5% / 5-95% / 95-100%）。
- 主线程预处理用 `_prep(label, frac)` 泵事件（`:938-950`，`QApplication.processEvents()` + 查取消标志）。
- `_sync_run_config()`（`:895-936`）把非标量写进 session → `ops.prepare(actor='gui', reason=…, diag=bool(self._god_mode))`（`:870-871`）
  → `_preflight_cost`（`:1255`，仅单体 mixed 且最远失谐 ≥3000 MHz 才弹成本报告；选"剔除远对"则写 `solve.COST_BUDGET=4e4` 并重建）
  → `RunWorker(ops, prepared, progress_arr, actor='gui')`（`evolve_worker.py:12`）`start()`。
- `RunWorker.run()`（`evolve_worker.py:39-55`）在工作线程只调 `ops.execute`（**不碰 cache**：ARC SQLite 线程亲和，`ops.py:147-151`）；
  结果经 `done(result, error)` 回主线程 → `_on_evolve_done`（`ui_dynamic.py:1412`）。
- 结果取回：`load_artifact(ops.root, artifact_id)`（`:1440-1445`）→ `_draw_bot` / `_draw_bot_pair`（人机同产物，I3 大载荷不内联）。
- Hub 模式分支（`:858-868`）：`HubRunWorker`（`evolve_worker.py:58`）→ `bridge.run()` = `run_compute`（Hub 侧 async）；
  **本地不 prepare/execute/load artifact**，绘图降级为只报 summary（`:1418-1427`）。

### 4.2 事故史对应的硬约束（**改动前必读**，都是实测事故）

| 约束 | 位置 | 事故 |
|---|---|---|
| 常驻进度控件**不得置 None** | `ui_dynamic.py:974-979`、`:814-816` | 曾算完停在 ~90-95%（引用被抹、没人再写） |
| 终态只由 `_finish_run_ui` 写 | `:1230-1234` | `QTimer.stop()` 不撤销已 post 的 timeout，迟到 tick 覆盖终态 |
| QThread 唯一强引用不能在线程还在跑时丢 | `:952-995`（`_retired_workers` + `finished` 释放） | 两体 dim=7056 跑完即 `0xc0000409`，结果全丢 |
| 槽里不抛异常 | 全 GUI（判空 + `except Exception`） | 槽内 `AttributeError` → `qFatal`/`0xc0000409`，无 Python 回溯 |
| 预处理期间 `processEvents` | `:938-950` | 长 V_dd 预处理时界面假死、取消点不动 |
| 成本报告用 `ApplicationModal` | `:1332` | 进度框 `WindowModal` 曾把成本报告压在下面点不到 |

### 4.3 当前运行流的已知缺口

1. **AI 模式点"运行演化"会静默失败**：`_sync_run_config` 的 `write_many` 无 try/except（`:936`），
   in-process AI 模式下 `authority.gate('gui')` 抛 `AuthorityError`（`mecha-sdk/mecha/session.py:126-127`），
   而 `_run` 只捕 `RuntimeError`（`:883`）⇒ 异常从 Qt 槽 `_on_run_clicked` 逃出（**qFatal 风险**）。
   （本轮未实测：GUI 无进入 in-process AI 模式的入口，见 §5。）
2. `_preflight_cost` 的"AI 不弹窗"分支是死码（`_ai_initiated` 从未置位，`:1307`）。
3. Hub 模式无进度/无取消（`HubRunWorker` 不传 `progress_arr`，`evolve_worker.py:69-79`），且 `_cancel_run` 只置本地标志（`:1212-1219`）⇒ Hub 侧运行取消不掉。

---

## 5. 两模式与写权（human/AI）——**GUI 侧只实现了一半**

### 5.1 语义（mecha 侧，权威）

- `Authority`（`mecha-sdk/mecha/authority.py:45`）：`can_write(actor)` = `side_of(actor) == mode`（`:59-61`）；
  `gate` 抛可教学 `AuthorityError`（`:63-73`）；`switch` 原子+记账+AI→人前要求 `in_flight==0`（`:75-103`）。
- `side_of`：AI 侧 `{ai, codeact, job, mcp}` / `job:<id>`；人侧 `{human, gui, user, session}`（`:26-42`）。
- 挂门：`Session(authority=…)`，`set/set_many/undo` 先过门（`session.py:126,137,167`）。
- `AppBridge.switch_mode`（`app_bridge.py:121-123`）；人工控制端点 = `src/adapter/serve.py:28-71`（`POST /switch_mode`，
  token 在 `.human-token`，固定 :8767，**非 MCP**——AI 不能自授权）。

### 5.2 GUI 侧实际有什么

| 项 | 状态 | 证据 |
|---|---|---|
| 写权门控**层** | ✅ 有 | `Binder._write`/`_write_external` 查 `can_write('gui')`（`_mecha_bind.py:69,238`） |
| 控件物理只读 | ❌ **未接线** | `set_readonly` 全库只被 `checks/gui_bind_selfcheck.py:59,73` 调用；`MainWindow` 从不调用 |
| 归因高亮 | ⚠ 仅 9 个绑定控件 | `_mecha_bind.py:109-127`；未绑定的键（波形/初态/能级组…）不刷新不高亮 |
| 总线实时刷新 | ⚠ in-process 有效 / Hub 模式失效 | `_mecha_bind.py:75-94` vs `hub_bridge.py:89-109`（形状不匹配，实测 AttributeError） |
| AI 流水面板（seq/actor/kind/target/before→after/reason） | ❌ **GUI 内不存在** | 全库无 `activity(` 的 GUI 调用点；bus 投影 UI 只在 web 驾驶舱（`dsh/src/client/CockpitTabBody.tsx:82-97` 飞行记录仪） |
| 写权切换入口 | ❌ **无** | `switch_mode` 无 GUI 调用点（grep：仅 checks + `serve.py` + `hub_bridge.py` 定义）；`HubBridge.switch_mode` 是死码 |
| 模式横幅 / overlay | ❌ 无 | 全库无 `banner`/`overlay`/`QDockWidget`（`sweep_gui_selfcheck.py:108-111` 反向断言"横幅类已删除"） |
| 状态栏 AI 提示 + 一键撤销 | ❌ **迁移时被删** | `git diff 25ff146^ 25ff146 -- app_entry.py`：删掉 `_init_ai()` 里的 `● AI 在线 :port` 指示、`撤销AI` 按钮、以及 `j.subscribe(_on_op)` 的 **"AI 来源 op 滚动到状态栏"** |
| 写权模式的可观测性 | ❌ 无 | 无 `authority_switch` 订阅者；`_hub_connected` 无人读 |

**结论**：`_mecha_bind` 把 I5 的"控件标记 + 实时刷新"做成了**可用的层**（并有 `gui_bind_selfcheck` 5 项守着），
但**没有接到 `MainWindow` 上**；"AI 写入即人可见（控件标记 + 流水 + reason）"这条纲领要求在 GUI 里**只完成了控件标记的机制、未完成接线，流水完全缺失**。
人夺权只能靠 `serve.py` 的 HTTP 端点（手工 `curl` 或外部脚本），GUI 内无路。

---

## 6. god-mode 检视器（`solver_inspector.py`，1282 行）

- **视图模式 ≠ 写权模式**：`MainWindow.set_mode(god=…)`（`app_entry.py:307-319`）只切"检视器显隐 + 标题 + 徽标 + 是否采 diag"；
  开关在菜单栏右上角互斥钮（`:281-305`）。默认极简。
- **常驻语义**：检视器**建一次不销毁**（`ui_dynamic.py:70-72,1357-1370,1386-1410`）；极简模式只 `hide()`，组件与绑定照旧存在。
- **6 个 tab**（`solver_inspector.py:299-312`）：处理链 / 哈密顿量 / 耦合对 / 求解过程 / 扇区与重构 / 引擎参数。
- **数据入口**：`load_diag(diag)`（`:400`）——`diag` 仅 mixed 引擎 `diag=True` 时产出；无 diag 时 `set_placeholder`（`:387`，
  占位文案见 `ui_dynamic.py:1404-1406`：hard/sambe 未接线）。摘要与 MCP/模型卡同源 `core.diag.summarize`（`:406-409`）。
- **引擎参数页**（`:465-605`）：数据源 `app_bridge.params(include_internal=True, page_size=100000)`（`:448`）；
  **`engine.*` 只读**（`readonly` 或 `family=='engine'` → `setReadOnly`，`:552,588-591`）、**`solve.*` 可写**
  （`editingFinished → _on_param_edit`，`:594,631-658`）→ `app_bridge.write(name, val, reason, actor='gui')`（`:646`），
  非法输入自愈回退会话现值 + 计数行红字（`:649-658`）。
- **diag 导出**：`save_diag_npz`（`:103-140`）+ `export_diag`（`:772-787`）。
- **窗口尺寸**：`resize(1500, 1080)`（`:275`，v1.2 注释解释为什么是 1080）。

### 6.1 检视器里的**死控件**（重要）

- `_bind_solver_knobs()`（`:325-337`，本应把 4 个"精度旋钮"设成只读）**从未被调用**（`_solver_knobs_bound` 仅在此函数内赋值，grep 无第二处）。
  ⇒ 处理链页的 `r_crit`/`per_freq_M`（`:806-815`）与求解页的 `tol`/`max_step`（`:1172-1181`）是**可编辑但无任何效果**的 QLineEdit
  （`_make_knob:315` 只 setPlaceholderText，无 connect）。
- `param_edited` 信号（`:262`）从不 emit；`set_god_params`（`:349`）/ `god_values`（`:381`）/ `param_overrides_valid`（`:664`）/
  `_LEGACY_KW2PATH`（`:367`）/ `_mirror_write`（`:339`）/ `_sync_mirrors`（`:660`）全是死码。
- `god_values()` 返回 `{'diag': True}`，但 `_run` 用的是 `diag=bool(self._god_mode)`（`ui_dynamic.py:871`）——
  即"上帝模式=采诊断"实际由**视图模式位**决定。
- 「⟳ 应用并重跑」（`:343-347`）只 emit `rerun_requested` → `_on_inspector_rerun`（`ui_dynamic.py:1379-1384`）
  = 确保上帝模式 + `self._run()`（**旋钮值不参与**）。

**改这块的原则**：要么把旋钮接回 `solve.*`（经 `_on_param_edit` 同一 choke point），要么**显式删除**并写清理由；
不要留着"能输入、没反应、也不报错"的控件（违反 D6/D5）。

---

## 7. 布局纪律与「布局红线清单」

### 7.1 铁律（决策 M3，出处 `docs/_history/engine-refactor/AI机甲框架-回归主仓库-迁移计划.md:18,134-140`）

> 控件**位置/尺寸/间距/排版/字体逐像素不变**；只允许精简**内部逻辑**。
> 新增 UI（横幅/高亮/流水）**不得回流既有排版**：横幅走状态栏/overlay，流水走 dock 或**检视器新 tab**，**绝不重排**。
> 验证手段原文要求"逐页截图 diff"——**注意：`ui_design/` 已在 M4 被删（commit 3b0dae4），仓库内今天没有现成截图/diff 工具**，需按 §10.3 自建。

### 7.2 红线清单（逐像素不能动；改这里必须走 §10）

**A. 主窗口骨架与自适应（最硬）**
- `app_entry.py:189-191` root/top 结构与 stretch；`:229` `control_widget` 的 `SizePolicy.Ignored × Preferred`；`:231` `setSpacing(2)`。
- `:247` `top.addWidget(canvas_tabs, 3)` / `:264` `top.addWidget(panel_scroll, 2)`（**画布:面板 = 3:2**，且画布在**左**）。
- `:255-257` `sweep_stack` 必须加在 `root` 里 `top` **之后**（"向上展开"的几何基础）。
- `:215-216` 状态栏 `addPermanentWidget` 的**个数与顺序**（`_run_label` 再 `_run_pbar`）。
- `:274` `QTimer.singleShot(0, _apply_fullscreen_layout)`。
- `:386-400` `_apply_fullscreen_layout` 的**全部系数**：面板宽 `0.22w ~ 0.34w`；谱表最小高 `max(140, 0.05h)`；
  雷达最小高 `max(150, 0.18h)`；`dpi = clamp(h/9, 100, 150)`。

**B. 画布几何**
- `ui_dynamic.py:106` `subplots_adjust(left=0.07, right=0.95, top=0.97, bottom=0.08)`（上下图时间轴对齐的依据）。
- `ui_dynamic.py:1507-1511` 双原子 `axA=[0.07,0.53,0.88,0.41]` / `axB=[0.07,0.06,0.88,0.41]`，单原子 `axA=[0.07,0.04,0.88,0.90]`。
- `ui_dynamic.py:1545-1551` y 轴 [0,1.02] + MaxNLocator(nbins=4) + 强制含 0；`:1575` `ax.margins(x=0)`。

**C. 硬编码尺寸（一个都不能改）**
- `fields.py:50,59` B/E `width=90`；`:75` `window_spin.setFixedWidth(52)`；`:87` `n_photon_spin.setFixedWidth(56)`；`:96` `add_btn.setMinimumWidth(110)`。
- `ui_dynamic.py:129,136,143` V_dd 三个 `width=128`；`:172` `frozen_h width=72`；`:183` `perfreq_m_edit.setFixedWidth(120)`；
  `:343` `decay_list.setFixedHeight(56)`；`:372` `readout_cb.setFixedWidth(300)`。
- `ui_group.py:55,63` `group_list`/`level_table` `setFixedHeight(180)`；`:61` 列宽 48；`:66` 行高 18；`:67` 表头高 36。
- `ui_spectrum.py:42` `spec_ref_edit.setMaximumWidth(110)`；`:52` `peak_pair_label.setFixedHeight(16)`；
  `:108,112,113,114,115,116` 列宽 240 / 55、最小高 140、最大高 200、行高 18、表头高 36。
- `solver_inspector.py:275` `resize(1500, 1080)`；`:841` 卡片标题 `setFixedWidth(150)`；`:850` 大数字 `setMinimumWidth(200)`；
  `:864` AC 表 `setFixedHeight(140)`；`:1072-1075` 耦合对表列宽 `(48,48,78,88,68,92)`。
- `hamiltonian_viewer.py:183` `t_label.setMinimumWidth(130)`；`:188` slider 最小宽 160；`:257` 色条 `setFixedWidth(36)`；
  `:278` hover 标签 `setFixedHeight(46)`；`:1486-1487` 色条最小 180×40。
- `sweep_panel.py:39-42` `MAX_ROWS=16 / BODY_MAX_H=190 / IDLE_HIDE_S=8 / POLL_MS=1000`；`:61` `SizePolicy.Expanding × Maximum`；
  `:74-75` margins `(8,3,8,4)` + spacing 2。
- `ui_theme.py:21-32` 字体与 9 个颜色常量；`:60-132` 整段 QSS（含 `QComboBox::down-arrow` 12×8）。

**D. 明确"不得回流"的位置**
- 横幅/告警：只能走 `statusBar()`（`addPermanentWidget` 或 `showMessage`）或 overlay —— 但**新增 permanent widget 也算改状态栏排版**，
  须先请示；`sweep_gui_selfcheck.py:110-111` 已把"主窗口上不得有 banner/window 属性"钉成判据。
- 流水/历史：只能走 **检视器新 tab** 或 dock（当前无 dock）。
- 归因高亮：**只能改 `setStyleSheet`**（`_mecha_bind.py:109-127` 的既有做法，存 `_mb_base_qss` 后叠加紫框），不得改几何。

### 7.3 自查配方（我本轮用过、可复现）

```powershell
# ① 改动 diff 里是否出现"布局类"调用（应为 0 或仅等价重表达）
git diff -- src/ui/ui_dynamic.py app_entry.py |
  Select-String '^[+-].*(setFixed|setSpacing|setContentsMargins|setMinimum|setMaximum|resize\(|setGeometry|setColumnWidth|setRowHeight|setStretch|setAlignment|font|StyleSheet)'
# ② 控件几何/文本 dump 前后对比（机器可读，不靠看图）
#    见 §10.3 的探针模板
```

> 实测示例（本轮）：迁移提交 `25ff146` 对 `src/ui/fields.py` 的 diff 里出现 3 条 `setFixedWidth`，
> 逐条核对为**等价重表达**（`pm.spinbox(..., width=52)` → 裸 `QSpinBox` + `setFixedWidth(52)`），像素不变。

---

## 8. 与纲领未对齐之处（I5 / M1 / M2 / M5 / D5 / D6）

| # | 纲领要求 | GUI 现状 | 证据 |
|---|---|---|---|
| 1 | **I5** AI 写入即人可见（控件标记 + **流水** + reason） | 控件标记机制✅（紫框 2s）/ 接线❌（未 `set_readonly`）/ **流水❌** | `_mecha_bind.py:109-127`；无 `activity()` UI |
| 2 | **I5** 人机同一条代码路径 | 运行路径✅（同一 `_run`/`ops`/artifact）；**写路径在 Hub 模式归因错**（人的写入记成 `ai`/`codeact`） | `ui_dynamic.py:795-882`；`server.py:94,148`；`codeact.py:106` |
| 3 | **I5** AI 路径不弹模态 | ⚠ 机制在但**没接线**（`_ai_initiated` 从未置位）；`refresh_plot` 与成本报告都是 `QMessageBox` | `ui_dynamic.py:1307,1322`；`app_entry.py:150` |
| 4 | **M1** 归因（actor+reason+provenance+reversibility+before/after） | 事件形状✅（`bus.py:182-193`）；GUI **只显示 reason 到状态栏**，无 actor/provenance 展示 | `app_entry.py:115-118` |
| 5 | **M2** 声称 vs 实测并排 | `session.set` 回执带 `readback`（`session.py:154-157`）但 **GUI 不显示**；只在 MCP 回程与 cockpit `/api/claim-measured` | — |
| 6 | **M5** 注意力分级 | 无分级：只有状态栏 `showMessage`（8s 超时）+ 若干 `QMessageBox.warning`（模态） | `app_entry.py:115` |
| 7 | **D5** 报错不静默 | ❌ 两处静默：`Binder._write` 门控吞写（`_mecha_bind.py:69-70`）；Hub 回程 `{'ok': False}` 无人检查（`ui_dynamic.py:936`） | — |
| 8 | **D6** 每个写入面配读回面 | `solve.*` 有读回（`_param_cell` 读会话现值 `:585-587`）；**「窗口/光子块」两个控件无写面也无读回**（见 §9-Q1） | `fields.py:75-94,260-263` |
| 9 | 两模式实时刷新 | in-process ✅ / **Hub ❌**（订阅形状不匹配，实测 AttributeError） | `_mecha_bind.py:63,76` vs `hub_bridge.py:89-109` |
| 10 | 夺权路径 | GUI 内无入口（只有 HTTP :8767 + `.human-token`）；`HubBridge.switch_mode` 死码 | `serve.py:62-71`；`hub_bridge.py:136-154` |
| 11 | 旧 P3「AI op 滚动到状态栏 + 一键撤销 AI」 | **迁移时删除**（`git diff 25ff146^ 25ff146 -- app_entry.py`，`_init_ai` 整段） | 未在新架构重建 |
| 12 | 大载荷 I3 / 人机同产物 | in-process ✅（artifact + `load_artifact`）；**Hub 模式降级为只报 summary**，人看不到曲线 | `ui_dynamic.py:1418-1427` |

---

## 9. 主代理五问 · 复核结论

### Q1 `docs/dsh插件-发布骨架.md` 说"GUI 全量移植、布局不变"——实际差异在哪？有没有偏离 M3？

**差异定位到提交**：迁移主提交 `25ff146`（"Re0重构覆盖父仓"），对 GUI 只动了 **6 个文件**：

| 文件 | 变化 | 性质 |
|---|---|---|
| `src/ui/_mecha_bind.py` | +287（新建） | 新层，取代 `pm_setup`/`param_manager` |
| `evolve_worker.py` | 瘦身（−190 量级） | 去掉 `_P.override`，改 `ops.prepare/execute` 拆分 |
| `solver_inspector.py` | 480 行改写 | 参数页改由 registry 建；engine.* 只读 |
| `ui_dynamic.py` | 464 行改写 | 状态改 session；加 Hub 分支（后续 `efa2af8`） |
| `app_entry.py` | −160 量级 | 组装 `AppBridge`+`Binder`；删 `param_reg`/`archive`/`_init_ai`/`engine_overrides` |
| `fields.py` | 42 行 | B/E 改绑 MB；window/n_photon 改本地 QSpinBox；加 `ac_kwargs()` |

**其余 9 个 `src/ui` 文件**（`ui_group`/`ui_spectrum`/`ui_radar`/`ui_single`/`ui_waves`/`ui_theme`/`hamiltonian_viewer`/`sweep_panel`/`app_state`）
在该提交**逐字节未动** ⇒ "布局不变"在文件层面成立。

**M3 偏离核查结果**：
- ✅ **几何层面无偏离**：diff 里的 3 条 `setFixedWidth` 是等价重表达（同宽同值同位置，§7.3 实测）。
- ⚠ **状态栏排版变了**（相对迁移前）：删掉了 `● AI 在线 :port` 指示与「撤销AI」按钮（`git diff 25ff146^ 25ff146 -- app_entry.py`）。
  这**恢复了更早的原始排版**，但按 M3 字面（"逐像素不变"）应记录为一次**有意删除**，别当成"从未存在过"。
- ❌ **一处功能性偏离（M1"旧模块基本无废止"）**：`fields.py` 的「窗口 / 光子块」两个控件
  从**注册键**（旧 `field.window` / `field.n_photon`）降级为**本地 GUI 态**，而 `ac_kwargs()`（`fields.py:260-263`）
  **全库无调用者**（grep 实测），`ui_single.py:119` 的 `build_level_rows(...)` 也不收 AC 精度参数。
  ⇒ 今天：**人改这两个框，什么都不会发生**（只触发一次重绘）；AI 也**改不动**——
  `AC_N_PHOTON` 是裸常量（`src/core/static_atom.py:523`，非 `Param`），注册键只剩 `engine.DIRECT_SOS_N_WINDOW`（`static_atom.py:61`，tier='run'）
  且 GUI 已与它脱钩。这是**静默失能控件**（D5/D6），也是"全量移植"里唯一被削掉的能力，建议尽快裁决：接回 registry 或删掉控件。
- ✅ 迁移后新增的 UI 都**没有回流既有排版**：进度条走状态栏（`app_entry.py:206-219`）、扫描栏走 `root` 尾部（`:255-257`）、
  高亮只改 stylesheet（`_mecha_bind.py:109-127`）。

### Q2 GUI 今天是 Hub 客户端还是自托管内核？怎么选？切换风险？

见 §1.3。要点：默认按 `.mcp-port` 自动选，失败 graceful 回落 in-process；`RE0_GUI_INPROC=1` 强制 in-process（判据统一设它）。
风险四条（订阅形状不匹配 / 归因错成 AI / 写失败静默 / 无进度无取消），另加"陈旧 `.mcp-port` 只造成一次快速失败"（实测 `_hub_connected=False`）。

### Q3 写权门控在 GUI 侧怎么体现？人怎么夺权？订阅总线做实时刷新与归因高亮做了多少？

见 §5。一句话：**门控只在 `Binder` 内部生效，且是"静默吞写"而非"控件禁用"**；`set_readonly` 从未被 `MainWindow` 调用；
人夺权**在 GUI 里无路**（只能外部打 `:8767` + `.human-token`）；实时刷新+归因高亮**只在 in-process 模式、且只覆盖 9 个绑定控件**。

### Q4 AI 流水面板在 GUI 里存在吗？

**不存在**。GUI 内无任何 bus 投影面板（无 `activity()` 调用、无 dock、无检视器流水 tab）。
唯一的流水投影在 **web 驾驶舱**：`dsh/src/client/CockpitTabBody.tsx:82-97`「飞行记录仪」页（逐行折叠：seq/actor/kind/reason，
展开 target/before→after/reason/reversibility），数据来自 cockpit HTTP `/api/history`（`:117`），3s 轮询（`:148`）。
另外迁移时删掉的旧实现（`_init_ai` 里的 `j.subscribe(_on_op)` 状态栏滚动）**没有在新架构重建**。

### Q5 旧耦合是否真清干净？

- `checks/cleanliness_guard_selfcheck.py:27-31` 的 `FORBID_IDS` = `pm_setup/engine_overrides/god_split/god_write/god_read/sim_state/_SOLVE_SPECS/register_panel_params/new_registry/_P/archive`，
  用 `tokenize` **只扫 NAME token、跳过 COMMENT/STRING**（`:52-53,66-70`）⇒ **注释里出现这些词不算命中**（`_mecha_bind.py:2,10`、`ui_dynamic.py:251`、`solver_inspector.py:327,626` 等处都是注释，故意保留）。
- 本轮实测：`run_checks.py --only cleanliness_guard` → **1 PASS / 0 FAIL**。`src/`、`checks/`、`app_entry.py` 零命中。
- **守卫的覆盖盲区（要记住）**：
  1. 只扫 `src/` + `checks/` + 根 `app_entry.py`（`:35-48`）——**不扫 `能级.py`、`scripts/`、`dsh/`**（我 grep 过 `scripts/`，零命中，无实害）；（原名单里的 `main.py` 已于 2026-09-25 删除，判据同步去掉）
  2. `'_archive' in dp` 整目录排除（`:39`）⇒ `src/_archive/*` 与 `checks/_archive/*` 里的旧名不算；
  3. `_P` 只在**模块别名**位置算脏，`self._P` 跳过（`:68-69`）；
  4. 只查这 11 个名字——**`src.ai` 不在名单里**（靠 `sim_state` 兜），而 `src/ai` 目录本身已删（`src/` 下无 `ai`）。
- 结论：**旧框架耦合在代码层面确实清干净了**（"残留"只出现在解释性注释与 `src/_archive/` 封存件里）。

---

## 10. 改界面的安全工序

### 10.0 前置（每轮开工前）

1. 确认**没有别的会话/进程在写这个仓库**（迁移计划 §8 已把"并发写锁"列为真实踩过的坑）。
2. 确认基线绿：`run_checks.py --only gui_`（6 判据 / 约 31s）。红着不要开工。
3. 读 §7.2 红线清单里**你要碰的那个文件**的全部条目。

### 10.1 施工（外科式最小 diff）

- 只动**逻辑**：信号连接、状态读写、异常处理、文案（tooltip/状态栏）。**不动**任何几何/字体/间距/QSS 结构。
- 新增 UI 只允许三种落点：**状态栏**（且新增 permanent widget 需先请示，它改状态栏排版）、**overlay**、**检视器新 tab**。
- 所有新槽函数：**入口判空 + `try/except Exception`**（Qt 槽抛异常 = `qFatal`/`0xc0000409`，无 Python 回溯）。
- 常驻控件引用**永不置 None**；QThread 引用只在 `isRunning()==False` 时丢。
- 写状态**只能**经 `app_bridge.write/write_many`（actor+reason 必填），不得直改 `session._snapshot`（I1）。
- 门控被拒时**必须可见**（状态栏提示），不要沿用 `_mecha_bind.py:69-70` 的静默 return（D5）。

### 10.2 验证（四道闸，按顺序）

```powershell
$env:QT_QPA_PLATFORM='offscreen'; $env:RE0_GUI_INPROC='1'; $env:PYTHONIOENCODING='utf-8'
# ① 实例化闸：MainWindow 能建起来
& D:\python310\python.exe -c "import app_entry; w=app_entry.MainWindow(); print('ok', type(w.app_bridge).__name__)"
# ② 布局闸：diff 里不得出现布局类调用（见 §7.3 配方）
# ③ 判据闸（含 sweep_gui，子串匹配）
& D:\python310\python.exe run_checks.py --only gui_     # 期望 scripts=6 fail=0
# ④ 收尾：涉及共享代码时请主代理跑全量（46/0，约 589s）——不要自行跑全量
```
（`os._exit` 前记得 `sys.stdout.flush()`：offscreen + matplotlib 的 teardown 会刷噪声并给非零退出码，
判据都用 `os._exit(0)` 绕过，见 `gui_app_entry_selfcheck.py:49-52`。）

### 10.3 几何 / 截图比对（M3 的可执行替代，因为 `ui_design/` 已删）

- **几何 dump（首选，机器可读）**：offscreen 建窗后遍历目标控件输出 `objectName/geometry/sizeHint/text`，
  改动前后各存一份（存 `$env:TEMP`，**不要往仓库写文件**）再 diff。
  模板可借 `skills/ui-design/shot_workflow_template.py`（**注意**：它的 `load_scenario` 用的
  `win.dynamic._restore_dynamic(...)` 已随迁移删除，改状态请用 `app_bridge.load(存档路径)`）。
- **截图比对**：`win.resize(1920,1080); win.show(); app.processEvents(); win.grab().save(png)`——
  本轮实测 offscreen 下 `grab()` 可用（`1400×800, isNull=False`）。
  ⚠ offscreen 虚拟屏只有 800×600，基于屏幕尺寸的自适应会被压扁 ⇒ 必须先 `resize` 再手工重设
  `panel_scroll` 的 min/max 宽（模板 `:59-63` 已给现成三行）。
  ⚠ 三条铁律（`skills/ui-design/SKILL.md:52-60`）：显式加载 CJK 字体（`C:/Windows/Fonts/msyh.ttc`，否则中文全是方框）、
  主模块 import 顺序（本项目实测 `import 能级` 必须在 `QApplication` 之前）、matplotlib `draw_idle()` 是异步的
  （`grab()` 前要 `canvas.draw()`）。
  离屏渲染是确定性的 ⇒ 像素级可比（`skills/ui-design/checklist.md:40`）。
- **逐元素核对**（`checklist.md:3-8`）：必须放大到元素级核对位置/对齐/可读性；只看整页缩略不算检验。

### 10.4 判据对照表（改哪儿跑哪儿）

| 改动面 | 判据 |
|---|---|
| 装配/入口/新面板 | `gui_app_entry_selfcheck.py`（实例化 + 关键部件在位） |
| `_mecha_bind` / 绑定 / 门控 / 高亮 | `gui_bind_selfcheck.py`（含**致死性**：人工能写 vs AI 模式被吞） |
| 检视器 / 参数页 / engine.* 只读 | `gui_inspector_selfcheck.py`（含"engine.* 全只读 / solve.* 可写"不变量） |
| 运行流 / `_sync_run_config` / 绘图回调 | `gui_run_flow_selfcheck.py` |
| `RunWorker` / QThread 包装 / artifact | `gui_run_worker_selfcheck.py` |
| 扫描栏（形态/收放/取消/人机同源） | `sweep_gui_selfcheck.py`（6 组，含反向断言"不是独立窗口""无横幅类"） |

**判据改动纪律**：`sweep_gui_selfcheck.py` 里有若干**用户裁决项**的反向断言（如 `:108-111`），
它们是"防回流"的钉子——**不要为了让新代码变绿而改判据**；先改设计。

---

## 附 A：本轮实测记录（可复现）

| 项 | 命令 | 结果 |
|---|---|---|
| GUI 判据子集 | `run_checks.py --only gui_` | `scripts=6 fail=0 seconds=31`（gui_app_entry 1 / gui_bind 5 / gui_inspector 3 / gui_run_flow 3 / gui_run_worker 2 / sweep_gui 6） |
| 扫描栏单跑 | `run_checks.py --only sweep_gui` | `scripts=1 fail=0 seconds=5`（6 PASS） |
| 清洗守卫 | `run_checks.py --only cleanliness_guard` | `scripts=1 fail=0 seconds=1` |
| offscreen 建窗 | `app_entry.MainWindow()` | 无异常；`hub_connected=False`、`bridge=AppBridge`、`binder._binds=9`、`_ext=0` |
| 截图能力 | `w.grab()` | `1400×800, isNull=False` |
| Hub 订阅形状 | `Binder._on_event(['field.B_gauss'])` | `AttributeError: 'list' object has no attribute 'get'`（见 §1.3 风险 1） |

## 附 B：文档声称 vs 代码实际（速查表）

| 声称 | 出处 | 实际 |
|---|---|---|
| "驾驶舱 iframe / 双模 native+iframe" | `dsh/README.md:125,129`、`docs/dsh插件-发布骨架.md:18,43` | 已删 iframe，**只留右栏 tab**（`dsh/src/client/index.ts:8,22,34-41`，commit `2e31ad4`） |
| "驾驶舱占主区、隐藏左栏" | `dsh/src/client/CockpitPanel.tsx:3-6,21-23` | 实际 `openRightbar + openTab('re0-cockpit')`（`index.ts:47,53`） |
| "app_entry 在后台线程跑进程内 MCP" | `mecha-sdk/mecha/app_bridge.py:10` | GUI 不调用 `mcp_server()`；Hub 是独立进程（`app_entry.py:446`） |
| "控制面板在左、画布在中央" | `docs/手册-界面·原理·开发.md:66-77` | 画布左、面板右（`app_entry.py:247,264`） |
| "存档 v3 登记册即存档" | 同上 `:80-81`、`§3.11` | 现行 `mecha-session-v1`（`app_bridge.py:127`） |
| "AI 也调 `self._run()`，故按钮/可取消性对两侧一致" | `ui_dynamic.py:1189-1190` | AI 走 MCP `run_compute`（Hub 进程），不经 `_run`；`_ai_initiated` 保护是死码 |
| "`ui_shot.py` / `layout_audit.py` 等成套工具直接调用" | `skills/ui-design/SKILL.md:64-84` | `ui_design/` 已删（commit `3b0dae4`，M4）；只剩通用模板与清单 |
| "`selftest_all.py --tier fast`" | 同上 `:91` | 现行 `run_checks.py`（`--only/--slow/--dir`） |
| "MCP 工具 `screenshot()` 拿当前主窗口 PNG" | 同上 `:79-80` | 18 工具清单里**无 screenshot**（`docs/README.md:59-62`） |

---

# 第 II 部分 · 多体页方案与界面侧数据管理（2026-09-22 第二轮，只读调研）

## 11. 多体页（`dynamic.mode=2`）界面方案

### 11.1 现状（代码事实）

| 事实 | 证据 |
|---|---|
| 引擎/契约已齐：`dynamic.mode` 枚举含 **2**，`manybody.*` 进 `config_schema`/`settable_keys`，`solve.*` 不膨胀 | `src/core/protocol.py:39-40,155-168`；`checks/manybody_integration_selfcheck.py:63-71` |
| 唯一分派已通：`build_run(mode=2)` → `runspec_mb`；`evolve_run` 走 `('ed','many')` | `src/core/runspec.py:497-502,90`；`protocol.py:298-392` |
| **GUI 完全无多体面** | `ui_dynamic.py` 只有两个单选（单原子/双原子，`:152-157`），`_on_mode_changed` 只认 `mode_id==1`（`:402`），`_sync_run_config` 只写 `'dynamic.mode': 1 if two else 0`（`:906`）⇒ **GUI 永远产不出 mode=2** |
| 页面注册机制 | `app_entry.py:239-247`：`modules` 表 7 元组 `(name, tab_widget, panel_widget, on_show, on_light, use_field, field_lights)`；循环 `canvas_tabs.addTab` + `page_stack.addWidget`。`_on_canvas_tab_changed(idx)` 用 `self.modules[idx]` 取元组（`:344-351`）⇒ **加第 3 项即自动适配，无需改切页逻辑** |
| `manybody.*` 实际是 **14 个键**（不是 12） | `src/core/manybody/schema_mb.py:28-72` |

**14 键的类型与"该配什么控件"**（`schema_mb.py` 行号）：

| # | 键 | 类型/枚举 | 需要的控件 | 备注 |
|---|---|---|---|---|
| 1 | `template` | str enum `ising/xy/potts3` | combo（**potts3 未实现** `runspec_mb.py:150-152` → 禁用+原因） | `:29-31` |
| 2 | `terms` | list，`I1/I2/I3` | **多选 checkbox 组**（现有工厂没有） | `:32-34` |
| 3 | `atoms` | `atom_list`（逐原子 `g/r` 四元组 + `pos` 三元组 + Ω/δ/Γ） | **★重控件：表格 + 点云画布**（真校验在 `validate_atom_list` `:92-123`） | `:35-38` |
| 4 | `psi0` | `mb_psi`：`None` / int / `'grg'` 模式串 / `'ground'` / `['sup',…]` | combo + 模式串输入（**`'sup'` 叠加编辑器后置**） | `:39-42` |
| 5 | `B_gauss` | float，`None`=复用 `field.B_gauss` | spinbox（**可留空**=复用） | `:43-45` |
| 6 | `n_window` | int ≥0 | spinbox | `:46-48` |
| 7 | `r_cut_um` | float，`None`=不截断 | spinbox（可留空） | `:49-51` |
| 8 | `energy_window_mhz` | float，`None` | spinbox（可留空） | `:52-54` |
| 9 | `solver` | str enum `auto/ed/tebd` | combo（**tebd 未实现** `runspec_mb.py:128-131` → 禁用+原因） | `:55-57` |
| 10 | `second_order` | bool | checkbox | `:58-60` |
| 11 | `guard_mhz` | float ≥0 | spinbox | `:61-63` |
| 12 | `ed_max_dim` | int ≥2（默认 2^20） | spinbox（int） | `:64-66` |
| 13 | `seed` | int，`None` | ⚠ **声明了但无消费者**：`runspec_mb.model_from_params` 不传 `seed`（`:74-82`），只有 `model.py:84` 读 `kw.get('seed')` | `:67-68` |
| 14 | `drive_segments` | list `[{t0,Om,delta},…]` | **★重控件：分段驱动表**（缺省=静态） | `:69-71`；消费点 `runspec_mb.py:199` → `engine_ed.py:196-231,302-304` |

**结果侧**：`evolve_manybody` 返回 `t/pops/states/pops_atom/S/corr/loss/error_report/diag`（`engine_ed.py:346-363`），
`result_labels`/`result_summary` 零改动可用（`checks/manybody_integration_selfcheck.py:144-153`）。
⚠ **但 `ops.save_artifact` 只落 `pops/t/loss/pops_A/pops_B/loss_A/loss_B`**（`mecha-sdk/mecha/ops.py:52`）
⇒ `pops_atom`/`corr`/`S` **过不了 artifact 边界**；GUI 走 `load_artifact`（`ui_dynamic.py:1440-1445`）只能拿到 `pops/t/loss`。
⇒ 逐原子 ⟨n_i⟩ 与 C_ij **可在 GUI 侧从联合 `pops` 重算**（`observables.atom_pops/corr_matrix` 只吃概率向量，`observables.py:30-42`），
但**纠缠熵 S 需要 psi，重算不了**（`:45-65`）⇒ 要显示 S 必须让 artifact 多落一个键（**引擎/mecha 侧改动，不在本模块**）。
另：`dim > _MB_FULLPOP_MAX` 时 `pops` 退化为逐原子边缘（`engine_ed.py:336-341`）⇒ 此时 C_ij 不可得，须在 UI 上写明。

### 11.2 方案（推荐：独立第 3 个 tab + 参数页进 `page_stack`）

**为什么必须进 `page_stack` 而不是追加到动态参数列**：实测（1920×1080，offscreen）
两页 `panel_scroll.verticalScrollBar().isVisible()` 均为 **False**（宽 272 / 内容 270）。
`page_stack` 是 `QStackedWidget`（按 tab **互斥占位、不累加**），而动态参数列是同一个 `QVBoxLayout`
（末尾两个 `stretch=1` 的波形容器会吸收/让出空间 ⇒ 追加控件会让既有控件**高度变化**）。
若参数变高到出现竖直滚动条，`group_panel`/`field_panel`（跨页共享）宽度会 **−10px** ⇒ **逐像素违规**。

**高度预算（硬）**：实测动态页 `page_stack` 高 **754px**（1080 屏）；多体参数面板须 ≤ 该值，否则滚动条出现。

### 11.3 文件级改动清单（可直接估工）

| 文件 | 改动 | 估行数 | 人日 |
|---|---|---|---|
| `src/ui/ui_manybody.py` | **新建** `ManyBodyPanel(QWidget)`：`manybody_widget`（参数，进 `page_stack`）+ 中央画布（上=点云/驱动，下=结果） | +700~900 | — |
| `app_entry.py` | `import`(1) + 建面板(1) + `modules` 加第 3 项(1)；**切页逻辑零改** | +3 | 0.1 |
| `src/ui/_mecha_bind.py` | 11 个标量键用**现有 5 个工厂**即可；新增 1 个 `checkgroup`（`terms` 多选）；`atoms`/`psi0`/`drive_segments` 走 `register_external`（该函数已存在、**当前全库 0 调用**，正是为这类非标参数设计的，`_mecha_bind.py:212-250`） | +40~60 | 0.3 |
| `src/ui/ui_dynamic.py` | **零改动**（独立 tab 方案）；若改走"动态页第 3 档"则需改 `_on_mode_changed`/`_sync_run_config` 写 `mode=2` | 0 / +15 | 0 / 0.5 |
| `checks/gui_manybody_selfcheck.py` | **新建**：建窗→切多体 tab→参数落 session(`dynamic.mode==2`)→`estimate` 走多体分支→`ops.prepare` 可跑→结果图不炸 | +80~110 | 0.5 |
| `src/core/**`、`mecha-sdk/**` | **不动**（除"artifact 多落 `pops_atom/corr/S`"需引擎侧裁决） | 0 | — |

**子任务与人日**：
M0 骨架（新 tab + 空面板 + 判据 + 参数页自动生成，14 键里 11 个标量）**1.5 天**；
M1 **★点云编辑器**（表格 + 画布点击加原子/拖拽 + 与 session 双向同步 + `validate_atom_list` 报错就地提示）**2.5–3 天**
（可复用现成交互范式：`ui_radar.py:79-80,200-227` 的 matplotlib 点击→回写参数）；
M2 **★结果图**（产物态布居 + ⟨n_i⟩(t) + C_ij 热图，从 `pops` 重算）**2 天**；
M3 ψ0 编辑器（None/int/模式串/ground）**0.5–1 天**；
M4 `drive_segments` 分段驱动表 **1.5 天（可后置）**；
M5 成本/可行性提示（`estimate()` 已现成：`dim/ed_feasible/est_sec/est_peak_gb`，`protocol.py:302-330`）**0.5 天**。
⇒ **合计 8.5–10 人日**；**M0+M1+M2+M5 ≈ 6.5–7 人日** 即可交付"能摆原子、能跑、能看关联"的可用页。

**可以先不做**：`drive_segments` 表、`psi0='sup'` 叠加编辑器、`template=potts3`/`solver=tebd`（未实现 → 控件禁用+写明原因，D3）、
`seed`（无消费者，见 §11.1 #13）、关联相图（2D 参数扫描应走 `src/sweep`，不要塞进本页）。
**多体不需要新的运行流**：`_run` 的 `ops.prepare/execute` + `RunWorker` + artifact 路径已通用（`ui_dynamic.py:870-882`）；
只需注意 `_preflight_cost` 对多体**直接返回**（`:1266` `method != 'mixed'` ⇒ 成本弹窗不触发），故 M5 的成本提示要由多体页自己调 `estimate()`。

### 11.4 会在哪里触碰 M3 红线 + 规避

| # | 触碰点 | 风险 | 规避 |
|---|---|---|---|
| 1 | `canvas_tabs.addTab` 第 3 页 | 既有 tab 不重排（QTabBar 左对齐、内容定宽），但**是新元素进入既有容器** | 需**用户裁决**（与 `sweep_stack` 同类先例）。若不许加 tab → 走"动态页第 3 档单选 + `setVisible(False)` 容器"（`ui_dynamic.py` 已有 `_vdd_box`/`_sambe_adv_box`/`decay_custom_box` 三个先例：隐藏时零高度 ⇒ 既有几何不变） |
| 2 | 右侧参数列高度 | 追加控件 → 出现竖直滚动条 → 共享面板宽度 −10px | **参数只进 `page_stack`**（互斥占位）；高度 ≤ 754px（1080 屏实测） |
| 3 | `_apply_fullscreen_layout` | 改系数/最小高度即违规 | **一行不碰**（`app_entry.py:386-400`） |
| 4 | 结果图画布位置 | 塞进右侧列会挤压既有控件 | 只放**中央画布区**（新 tab 内），右侧不新增控件 |
| 5 | 告警/高亮 | 新增 permanent widget 改状态栏排版 | 走 `statusBar().showMessage()` 或 stylesheet（§7.2-D） |

---

## 12. 界面侧"数据管理"现状（GUI 可读的持久数据 + 清理入口）

### 12.1 清单（含体积与清理入口）

| # | 位置 | 量 | 谁读 | 清理入口 | 备注 |
|---|---|---|---|---|---|
| 1 | `save_state/*.json` | **12 份 / 0.6MB 内** | GUI：文件→保存/恢复（`app_entry.py:269-273,363-384`）；AI：`list_archives`/`read_archive`（`mecha-sdk/mecha/archives.py:41,63`） | **GUI 无**（只有保存/恢复对话框） | 格式全为 `mecha-session-v1`（实测 12/12 含 `format`） |
| 2 | `save_state/*.v2bak` | **12 份** | **无任何代码读**（grep 仅文档） | 无 | M5 一次性转换的备份（commit `25ff146`）；`精确光场.json.v2bak` 97KB vs 现行 628B |
| 3 | `save_state/_autosnap/*.json` | **20 份**（2026-09-11~16） | **无**（写者随 `src/ai` 删除；grep 无代码引用） | 无 | **孤儿目录**：手册 `:374,1427` 称"保留最近 20 份"已失效（无轮转代码） |
| 4 | `save_state/_legacy/*_v1.json` | 2 份 | **无**（`checks/` 里 grep `save_state` **零命中**） | 无 | 手册 `:1424` 称"v1 兼容回归样本"，**无判据读它** ⇒ 文档声称无实现支撑 |
| 5 | `runtime/ai_results/` | 151 文件 **43.8MB** | AI 面（`SKILL.md` 所述 job 落盘） | **无** | gitignored |
| 6 | `runtime/sweeps/` | 463 文件 **32.7MB** | `list_sweeps`（`src/sweep/registry.py:362-387`，最近 20） | **无**；**GUI 不调 `list_sweeps`**（`sweep_panel` 只读当前 `status()`） | gitignored |
| 7 | `runtime/_journal/*.jsonl` | 7 文件 0.89MB | 无（旧 AI journal） | 无 | ⚠ **唯一被 git 跟踪的运行数据**（`.gitignore` 未列） |
| 8 | `.mecha/journal.jsonl` | 0.08MB | `Bus`（`bus.py:137-153`）+ cockpit 只读 | **无 prune**；`_maybe_rotate` 8MB 才轮转（`bus.py:159-169`） | gitignored |
| 9 | `.mecha/artifacts/` | 10 目录 0.18MB | GUI `load_artifact`（`ui_dynamic.py:1443`）；`ops.rerun` | **无 prune** | 内容寻址（同配置同 aid）⇒ 天然去重 |
| 10 | `.mecha/payloads/` | 2 文件 | `bus` 指针/截断回读 | **无 prune** | 同上 |
| 11 | `.arcdata/` | 25 文件 **67MB** | ARC（引擎） | 无 | gitignored |

**总量**：`runtime/` 77.4MB + `.arcdata/` 67MB + `save_state/` 0.6MB + `.mecha/` 0.28MB。

### 12.2 结论

1. **GUI 侧清理入口 = 0 个**：全库 `os.remove` 只出现在 `src/adapter/serve.py:119`（退出删 `.mcp-port`）。
   连"列出存档"都只有 AI 面（`list_archives`），GUI 只能靠 `QFileDialog` 目视。
2. **老旧数据确实在堆积，且大部分已无消费者**：#2 `*.v2bak`、#3 `_autosnap/`、#4 `_legacy/`、#7 `runtime/_journal/` 都是孤儿（合计 34 个文件）。
3. **两类"格式门"行为不同（都算"报错不静默"，但提示不准确）**：
   - `_autosnap/*.json` 含 `params`（58 个**迁移前**键）⇒ 过得了 `app_bridge.load` 的格式门（只查 `'params' in snap`，`mecha-sdk/mecha/app_bridge.py:142-145`），
     但 `set_many` 全校验会因未知键抛错（`session.py:139`）⇒ 人看到的错误是"未知键"而非"旧格式"，容易误判为程序 bug；
   - `_legacy/*_v1.json` 无 `params` ⇒ 直接报"不是 mecha-session-v1 存档"（准确）。
4. **检视器/流水不展示历史**：`load_diag` 只在内存保留最近一次（`ui_dynamic.py:1401-1402`）；无流水面板（§5）；无历史产物列表。
5. **建议落点（守 M3）**：清理功能**不要**进既有布局——放**文件菜单**新增项（QMenu 项不占布局，`app_entry.py:269-273` 同一处）
   或**检视器新 tab**（"存档/产物"页：列表 + 体积 + 删除，删除前先归档）。
   `checks/cockpit_selfcheck.py:178` 已把"cockpit 不得 remove/unlink/rmtree"钉成判据（黑匣子纯净）——清理必须归 GUI/人，别放进 cockpit。

---

## 13. 优先级建议（D1–D3 + 多体页）

| 序 | 项 | 工作量 | 为什么排这个位置 |
|---|---|---|---|
| **P0** | **D3 死接线清理**（检视器 4 个精度旋钮、`param_edited`/`set_god_params`/`god_values`/`_LEGACY_KW2PATH`、`_ai_initiated`、`_show_progress` 死分支） | **0.5 天** | 零布局风险；消除"能输入没反应"（D5/D6）；且它们是改界面时的**噪声源**（易被当成活代码）。`gui_inspector` 判据已覆盖参数页 |
| **P1** | **D1 `field.window`/光子块失能** | **1–1.5 天** | 唯一被削掉的能力。**建议"接回 registry"而非"删控件"**：删控件会改那一行几何（更贵且碰 M3）；接回只改参数传递（`fields.ac_kwargs()` → `ui_single.draw_levels`/`ui_spectrum.refresh_view`），需重跑静态页判据 |
| **P2** | **D2 两模式门控接线**（`set_readonly` + 订阅 `authority_switch` + 修 Hub 订阅形状 + 状态栏提示） | **2–3 天** | 纲领 I5/§5 硬要求，**纯逻辑无几何**；流水面板单列一档，落**检视器新 tab**（M3 明文允许） |
| **P3** | **多体页** | **6.5–10 人日** | 增量能力、不影响既有正确性，但工作量最大且需先裁决"第 3 tab vs 第 3 档" |
| **P4** | 数据管理清理入口（文件菜单或检视器新 tab） | 1–1.5 天 | 与 P0/P2 同属"接线类"，可在 P2 之后顺手做（都落检视器/菜单，零布局风险） |

**理由一句话**：P0–P2 是**已交付功能的正确性缺口**（有判据、有纲领条款、有实测证据），多体页是**新增能力**；
先把"已有的别骗人"补齐，再铺新面。P0 尤其该先做——它顺手把改界面的"安全工序"变干净。

---

> **维护约定**：本手册随 GUI 改动同步更新；每次改动请回填「附 A 实测记录」与相关 `文件:行号`。
> 若发现本文与代码不符，**以代码为准**并在「附 B」登记。
