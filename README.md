# 原子能级计算器（Rb-87 / 通用碱金属）

一个面向里德堡原子实验的**原子能级 + 寻址光 AC-Stark + 单/双体跃迁频谱**计算与可视化工具。界面可直接交互：实时调整磁场 / 电场 / 光场，直观看到能级位移、超精细结构、谱线变化，并通过点击谱峰定位跃迁对。

> 本项目以 Rb-87 为默认原子，但物理核心与 ARC 库解耦，**可切换到任意 ARC 支持的碱金属原子**（Cs、Li、Na、K…）。

---

## 项目结构

```
Energy Level/
├── 能级.py           入口（GUI，PyQt5 + matplotlib；内部调用 src 包）
├── src/              核心包（无 UI 依赖的引擎 + UI 控件）
│   ├── core.py       单原子物理核心（能级 / AC/DC-Stark / 偶极矩阵元 / 谱线）
│   ├── pair_core.py  两体静态核心（pair 哈密顿量 / vdW / 交换耦合 / 本征分解）
│   ├── driver.py     统一引擎（单体/双体共用：构建系统 → 驱动 → 响应谱 / 演化）
│   ├── ui_single.py  旧版单体画布（EnergyPlotPanel：能级图 + off/on 镜像频谱）
│   ├── ui_pair.py    响应谱画布（ResponsePlot）与响应谱曲线计算
│   └── ui_fields.py  共用控制面板（磁场/电场/光场表格/关注跃迁/初态）
├── checks/           回归自检（selftest_all.py 一键全跑）
│   ├── checklib.py / core_selfcheck / core_ac_directcheck
│   ├── pair_selfcheck / unified_selfcheck
├── scripts/          辅助/校验脚本（compare_pairinteraction、ac_shift_curve、demo_*）
└── _misc/            探索原型与杂物（两体脚本、设计文档、VS Code 主题）
```

**架构要点**：单体和双体共用一套引擎与操作面板（`driver.build_system` 以 `n_body=1/2` 区分），
切换的只是左侧画布（单体页 / 双体页）与触发的计算。物理核心（`core`/`pair_core`）无 UI 依赖，可单独审查与复用。

---

## 一、功能总览

| 能力 | 说明 |
|---|---|
| 能级结构 | fine 结构 + 超精细 + Zeeman（B 场任意），数值对角化 |
| AC-Stark | 近共振 dressed 精确解 + 远失谐非 RWA SOS，失谐精确到子能级（含超精细/Zeeman） |
| DC-Stark | 二阶微扰 SOS（含超精细/Zeeman 子能级），E² 标度已验证 |
| 两体相互作用 | pair 哈密顿量本征分解、vdW 能移、交换耦合、`R/θ/φ/m_total` 几何配置 |
| 跃迁频谱 | 全部偶极允许跃迁，爱因斯坦 A 系数强度，偏振分辨 |
| 响应谱 | 黄金规则线性响应谱（单体/双体共用引擎），共振峰 + 峰表 |
| 两能谱对比 | exchange 模式：一块画布 off 谱向上、on 谱向下——肉眼发现交换相互作用 |
| 光场表格 | 多光管理：频率 / 光强 / 偏振 / 作用范围（both / atom1 / atom2），右键删除 |
| 关注跃迁 | 下拉选择关注哪个光场（或 exchange）→ 决定能移计算与响应谱绘制 |
| 回归自检 | `core_selfcheck.py`（单原子 24 项）+ `pair_selfcheck.py`（17 项）+ `unified_selfcheck.py`（18 项） |

---

## 二、安装与运行

**环境**：Python 3.10+（实测 3.10）。**依赖**：

```
numpy  scipy  matplotlib  PyQt5  arc
```

（`arc` 为 [ARC — Alkaline Rydberg Calculator](https://arc-alkali-rydberg-calculator.readthedocs.io/) 库，提供碱金属量子亏损、波函数、偶极矩阵元等。）

**运行**（在项目根目录内）：

```bash
python 能级.py            # 启动 GUI
python checks\selftest_all.py   # 一键跑全部回归自检
```

> Windows 环境如 `python` 解析异常，用绝对路径：`& "D:\python310\python.exe" 能级.py`

---

## 三、使用说明书

### 3.1 界面布局

- **左侧画布**（QTab 切换）：
  - **单体页**：`响应谱`（黄金规则线性响应谱 + 峰表联动）与 `能级图`（旧版：off/on 两能谱镜像对比）
  - **双体页**：两体响应谱（含 vdW/交换相互作用，peak 点击高亮）
- **右侧共用面板**：
  - **磁场 B (G)** / **电场 E (V/cm)** / **添加光** 同一行
  - **光场表格**：序号 / 频率 (GHz) / 光强 (mW/cm²) / 偏振 / 作用范围（both / atom1 / atom2），右键删除
  - **关注跃迁**：下拉选择关注哪个光场（列出已创建光场；单体页额外含 exchange）
  - **初态**：|n l j m⟩ 输入
  - 能级组 / 显示范围 / 参考线 / 谱横轴范围 等控件

### 3.2 基本流程

1. **添加能级组**：右侧输入 `n / l / j`（如 60/0/0.5 添加 60S），点"添加"；重复可添加多个组（6P、60P…）
2. **设置磁场**：`B (G)` 输入框（回车生效），如 21.7 G
3. **添加光场**：点"添加光"，在表格里填频率（GHz）/ 光强（mW/cm²）/ 偏振 / 作用范围
4. **选择关注跃迁**：下拉选择要研究的跃迁（光场或 exchange）→ 决定能移计算与响应谱绘制
5. **观察**：左侧画布看响应谱 / 能级图 / 两能谱对比；点击谱峰高亮并联动峰表
6. **交换相互作用**（单体页关注跃迁选 exchange）：画 off/on 两能谱镜像对比

### 3.3 常见使用提示

- 想观察"某对能级在共振下的大位移"：**偏振方向决定哪侧共振**。例：σ⁻ 下 6P3/2 F=3 mF=3 ↔ 60S mJ=0.5 是 6P3/2 侧大位移；要让 60S mJ=+0.5 大位移请用 σ⁺（60S 0.5 →σ⁺→ 6P3/2 1.5）。
- B=0 时超精细 F 内各 mF 简并，无法从频率区分同 F 不同 mF 的跃迁；B≠0（默认 21.7 G）才可区分。
- 频率框为 3 位小数 GHz（约 1 MHz 精度），亚 MHz 失谐会被舍入。
- 共振光若同时作能移会把自己 dressed 推走（0 峰）——只对"关注跃迁"光作驱动，其余作能移。
