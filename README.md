# Rb-87 里德堡原子仿真软件

> **⁸⁷Rb 里德堡原子的静态计算 + 含时演化桌面软件**（PyQt5，Python 3.10）。
>
> - **静态**：Zeeman / DC-Stark 分裂、寻址光 AC-Stark（光）频移、跃迁谱线（频率/偏振/强度）、
>   耦合强度、双原子偶极-偶极（V_dd）耦合的角度/距离依赖。
> - **含时**：微波 + 光脉冲驱动下的含时薛定谔方程演化——π 脉冲、Rabi 振荡、STIRAP、
>   双原子共振能量交换（Förster 交换）。这是软件最核心的部分。
> - **三个自研引擎**：`sambe`（光子块，无 RWA）/ `hard`（旋转系 + 微波硬积分）/
>   `mixed`（φ-帧：光当粒子、微波当场，5 倍降维）。原子数据来自 ARC，
>   演化内核全部自主实现。
> - **可被 AI 驱动**：软件内建 MCP server，任何支持 MCP 的 AI 能像人一样操作它
>   （人机同一状态对象、同一语义入口，每次变更可追溯、可撤销）。
>
> 📖 **详细内容在 `docs/`**（本文只做入口）：界面操作 / 三引擎原理 / 开发指南全文见
> **`docs/手册-界面·原理·开发.md`**；历史任务报告见 `docs/历史-任务报告汇编.md`。

---

## 1. 怎么跑

```bash
pip install -r requirements.txt
python scripts/setup_arcdata.py      # 可选：准备 ARC 原子数据（缺省用 ~/.arc-data）
python 能级.py                        # 日常开发（等价：python app_entry.py）
```

| 命令 | 场景 |
|---|---|
| `python 能级.py` | 日常开发 |
| `python app_entry.py` | 直接运行（自带 `__main__` 守卫） |
| `main.py` / `dist/`（PyInstaller） | 分发给无 Python 环境的用户：`build.bat` |

启动后主窗口：左侧可滚动控制面板（能级组 → 场 → 页面参数），中央 **「静态」/「动态」** 两页，
菜单 **文件 → 保存状态 / 恢复状态**（存档 v3「登记册即存档」，v1/v2 旧档自动迁移）。

## 2. 跑自检（改代码后必做）

```bash
python checks/selftest_all.py --jobs 4        # 默认档（full）+ 并行 4 ⇒ 约 2 分钟
python checks/selftest_all.py --tier fast     # 提交前快筛（更小子集）
python checks/selftest_all.py --tier all      # 连"重对拍"档一起跑（发版/动过引擎时）
python checks/selftest_all.py --verbose       # 展开每个不变式包的子条件
python checks/checks_sentinel.py              # 只看存活哨兵（秒级）
```

- **三个档**：`full`（默认——哨兵/纯函数/单体静态/轻 GUI/引擎端到端不变式）、
  `fast`（快筛）、`deep`（五个"引擎算得对"的**权威对拍/长演化**体检，约 4 分钟，
  **按需**：它与默认档的 `engine_e2e_selfcheck.py` 覆盖同一批物理不变式，但保留了
  各引擎的解析解/外部权威包交叉验证）；`--tier all` = 全部。
- **计数口径 = 不变式包**：`Checker.group()` 把同一不变式的十几条碎片断言合成 **1 项**，
  包内**子条件**（`SUB`）单独累计 ⇒ 摘要给的是"有多少件物理"，而不是"有多少行断言"；
  包失败时子条件自动展开（`--verbose` 全展开）。
- **并行**：脚本之间无共享状态，`--jobs 4` 实测默认档 **128–135 s**、含 deep 档
  **165 s**（串行分别 358 s / ~520 s）；**Qt/QThread 时序类**
  （`gui_worker_lifetime_test.py`，0xc0000409 事故的唯一守卫）被标为 `_SERIAL_ONLY`
  ——**先串行跑完**再并行其余，保证确定性（2026-09-14 使用者抓到它在 `--jobs 2` 下
  偶发假红 ⇒ 已从"靠 sleep 猜时序"改成**事件驱动**：worker 阻塞到测试放行，
  结束时刻由测试决定；并补了反向守卫"仍在跑的 worker 绝不能被 sweep 丢掉"）。
- **实测基准**（本机，2026-09-14）：默认档 `full --jobs 4` = 31 脚本 / **183 项** /
  **661 子条件** / 0 FAIL / **128–135 s**（连跑 3 次）；`--tier all --jobs 4` =
  36 脚本 / 238 项 / 716 子条件 / 165 s；串行 `--jobs 1` = 358 s。
- 自检**自动带** `EL_NO_AI_SERVER=1` + `QT_QPA_PLATFORM=offscreen`：不抢活实例的 8765
  端口、不弹窗、不覆盖 `.mcp-port`（2026-09-12 事故后立的规矩）；另有**单脚本看门狗**
  （默认 600 s，超时杀进程并记 FAIL，防挂死拖住并行档）。
- 跑完给一张按档分组的**汇总表**（每脚本：耗时 / 项数 / 子条件 / FAIL / SKIP）+ 一行
  机器可读的 `##SELFTEST_SUMMARY scripts=… items=… subs=… fail=… seconds=…`。
- **权威清单**（脚本 → 覆盖点 → 项数）在 `checks/selftest_all.py` 的模块 docstring，
  与 `_PLAN`/`_DEEP`/`_SERIAL_ONLY` 同文件维护；按需单跑与 `checks/_archive/` 的说明也在那里。
- **哨兵**（`checks/checks_sentinel.py`）最先跑：语法 + `src.*` 导入可解析 + 兼容导出
  未遮蔽子包 + 同目录隐式依赖登记（连"文件带 UTF-8 BOM"这类文本卫生也会被它抓出来）。

## 3. 界面速览（详细操作 → 手册 §1）

| 区域 | 一句话 |
|---|---|
| 能级组管理（两页共用） | `n/l/j` + 添加；左栏彩色列表**右键**「子能级…」裁剪、「删除此能级组」 |
| 场面板 | 磁场 B、电场 E、光场表（频率/强度/偏振）；地址光与损耗设定 |
| 静态页 | 能级图 + 镜像频谱、耦合强度雷达图、V_dd 角度/距离扫描 |
| 动态页 | 单/双原子模式、波形序列（含任意波形文件导入）、引擎档位、初态与自发辐射、运行/进度/取消 |
| 上帝模式检视器 | mixed 求解链可视化（H(t) 快照、分项矩阵、扇区、收敛诊断）；窗口缩放 + 只读视图 |
| 状态栏 | 所有计算日志 / 路由理由 / AI 操作流水的统一出口 |

## 4. 让 AI 驱动（MCP）

```bash
pip install "mcp>=2,<3" uvicorn     # 可选依赖；缺失时软件照常运行，仅提示 AI 未启用
```

软件启动即内建 MCP server（streamable HTTP，默认 8765，实际端口写入 `.mcp-port`，
状态栏常驻「● AI 在线 :端口」）。任何 MCP harness 配上 `http://127.0.0.1:8765/mcp`
即可驱动：AI 改参数/点计算，GUI 实时反映（控件闪紫框）；人可随时点「撤销AI」回滚。

- 工具面：状态读写 / 参数 / 能级组 / 波形 / 计算与作业 / 流水与撤销 / 存档 / 截图 /
  **`run_python`（CodeAct，多步编排一次调用跑完）** /
  **`submit_sweep`（并行扫描，见下）**——完整清单见手册 §1.9。
- **并行扫描（2026-09-15 加）**：`submit_sweep(axes={…})` 一次调用把整个参数网格交给
  **独立 runner 进程的多进程池**（`src/sweep/`），立即返回 `sweep_id`；
  `sweep_status` / `sweep_result` / `cancel_sweep` / `list_sweeps` 跟进。
  轴用**登记册路径**表达（`'field.lights[0].intensity'`、`'dynamic.waves_A[0].env'`、
  `'dynamic.t_max'`），支持 `{'linspace':…}` 等形态并自动做笛卡尔积；并行度按
  **核数 + 可用内存**自动定档并给出理由。人侧在右下角有**逐点进度悬浮窗**、
  画布顶部有一行「⚡ 并行扫描运行中」，两者与 AI 读**同一份** `status.json`。
  设计与验收见 `docs/并行扫描-设计与验收.md`。
- **给 AI 看的说明书**：`skills/energy-level/SKILL.md`（工具语义、单位约定、波形格式、
  20+ 条已知坑）。AI 使用中发现不符可回头修正（自我迭代）。
- 人机协同协议与意见登记区见 `docs/AI协同-分工与请示.md`。

## 5. 用脚本直接调引擎

```python
from src.core.evolve_engine import evolve_engine      # 统一入口（sambe/hard/mixed）
res = evolve_engine('hard', states, fields=fields, psi0=0,
                    t_grid=t, n_body=1)               # → dict(t, pops, states, …)
```

数据模型（`states` / 三种场格式 / `Wave` / `psi0` 三形态 / 单位约定）、返回契约、
两体脚本写法与静态 API 速查见 **手册 §3.2–§3.7**（含可直接抄的最小示例）。

## 6. 去哪儿看

**`docs/`（18 篇）**

| 文档 | 用途 |
|---|---|
| `手册-界面·原理·开发.md` | **界面操作 + 三引擎推导 + 开发指南**（原 README 正文，逐字迁入） |
| `引擎参数总表.md` | 全库硬编码参数的**唯一定义点清单**（值/位置/依据） |
| `核查报告-文档与代码不一致与整改项.md` | 三引擎「文档 ↔ 代码」一致性核查（12 项整改 + 实测复现命令 + 已排除误报） |
| `非幺正演化-自发辐射与耗散模型.md` | 自发辐射/耗散/纯退相怎么建模：非厄米 vs Lindblad 两套数学、逐引擎接线、loss/norm 精度口径、可达性陷阱 |
| `场致加热与损失-调查与建模方案.md` | 场致损失/反冲加热的调查 + 分层建模方案（含实测：现有 `scatter` 公式在里德堡寻址几何下取反） |
| `引擎能力边界-终稿.md` | 三引擎适用域/局限/路由结论 |
| `混合引擎理论推导-光Sambe微波当场.md` | φ-帧 + AC 化（现行理论） |
| `M选择指南-全频段两体Sambe.md` | 逐载频光子数 M 的选择与性能档 |
| `auto路由方案-动态引擎选择.md` | `route_auto` 判据（引擎层 API） |
| `参数与判据决策史.md` | 参数/判据的历史动因与实测锚 |
| `freeze-纠缠态验证与基组收敛.md` | 当前主线：freeze 打断交换 + 基组收敛事故 |
| `freeze优化-AutoTuner交接.md` | freeze 黑盒/锚点/约束（交接给 AutoTuner） |
| `叠加态初态-接口设计.md` | 逐原子/两体叠加态初态接口 |
| `引擎调用与多原子扩展-待办.md` | 引擎调用摩擦点 + 多原子扩展待办 |
| `AI协同-分工与请示.md` | 两个 AI 角色的分工协议 + 意见登记区 |
| `AI协同-项目目标与原则.md` | AI 接入的顶层原则（G1–G4 / P1–P6） |
| `上帝模式-AI接入设计.md` | 上帝模式/检视器的 AI 可及面 |
| `历史-任务报告汇编.md` | **历史只作历史**：33 条，每条 4 行 + `git show` 定位原文 |

**代码**

| 位置 | 内容 |
|---|---|
| `src/core/` | 引擎与物理：`sambe.py` / `hard.py` / `evolve_mixed.py` / `evolve_twofield.py` / 静态（`static_atom.py` / `static_pair.py`）/ 场（`field_build.py`）/ 参数注册表（`params.py`） |
| `src/ui/` | PyQt5 界面：`ui_dynamic.py`（动态页）/ `ui_group.py` / `fields.py` / `solver_inspector.py`（检视器）/ `evolve_worker.py` |
| `src/ai/` | AI 接入：`server.py`（MCP 工具）/ `sim_state.py`（语义状态门面）/ `journal.py`（操作流水）/ `codeact.py` |
| `src/archive.py` | 存档 v3（登记册即存档） |
| `checks/` | 自检（见 §2）；`checks/_archive/` = 已封存、按需单跑 |
| `skills/energy-level/` | 给 AI 的操作手册（SKILL） |
| `scripts/` | 维护/交叉验证脚本（`setup_arcdata.py` 准备 ARC 数据、`make_down_arrow.py` 生成 UI 资产、`compare_pairinteraction*.py` 与外部权威包对拍、`slim_journal.py` 迁移历史流水） |

## 7. 已知问题与环境限制

| 问题 | 现状 / 处置 |
|---|---|
| **任意波形 + 含光学流形的基组 → 极慢** | 任意波形走 `mw_group` 路径、不参与光场旋转 → RHS 带 295 THz 相位，步长被钉死（实测同一 2 ns：无 6P 时 15.9 s，含 6P 时 60 s 未完成）。运行前会打印适用性警告（`field_build.arb_applicability`）；光学驱动请改用光场波形或按"光强包络"导入 |
| **长演化偶发 Qt/ARC 堆损坏** | 全流程长演化偶发崩、隔离单跑正常 → 判为环境/ARC 交互；自检把长演化放独立进程跑 |
| **受限沙箱里删文件会失败** | 安全删除策略 fail-closed：临时文件一律写系统临时目录 + 删除套 `try/except OSError` |
| **AI 端点未启用** | 缺 `mcp`/`uvicorn`，或 `EL_NO_AI_SERVER=1`（自检模式），或端口被活实例占用（此时**拒绝抢端口**并提示）|

## 8. 本次瘦身与自检重设计（2026-09-14，只动文档/草稿区/自检，不动物理）

| 对象 | 前 | 后 |
|---|---|---|
| `README.md` | 90.8 KB / 1445 行（使用+原理+开发三部分） | 本文件（入口），正文逐字迁入 `docs/手册-界面·原理·开发.md` |
| `docs/` | 45 篇 / 449 KB | **15 篇**；32 份历史报告 → `历史-任务报告汇编.md`（33 条 × 4 行，`git show` 可取原文） |
| `checks/` 计数口径 | 677 项（= 断言条数，碎片化） | **183 项**（不变式包）+ **661 子条件**（覆盖面）；`--tier all` = 238 项 / 716 子条件 |
| `checks/` 耗时（默认档） | 串行 635 s | **串行 358 s**；`--jobs 4` **128–135 s**（连跑 3 次 0 FAIL）；含 deep 档 165 s |
| `checks/` 结构 | 32 脚本一档 | 31 脚本 + **三档**（`fast`/`full`/`deep`）+ `--jobs N` + 单脚本看门狗；新增 `engine_e2e_selfcheck.py`（7 包 / 25 子条件 / ~30 s）收敛五个重对拍（deep 档保留、不删） |
| `_journal/` | 39.8 MB（单条 detail 曾达 748 KB） | **0.83 MB**；写入侧加 detail 体积闸（超限只存形状/长度/sha1）+ `checks/journal_size_cap_selfcheck.py` 21 项 |
| `scripts/` | 11 个（含 3 个改动前就已腐坏、3 个一次性产物） | 6 个（删腐坏/产物，加 `slim_journal.py`） |
| `tempsimu/` `ai_results/` `ui_design/screenshots/` | 1301 / 126 / 71 个文件 | 34 / 0 / 6（由使用者侧清理，仓库外留有打包件） |

> 物理与引擎代码在这两轮**未做任何改动**（`src/` 的改动只有 `src/ai/journal.py` 的
> detail 体积闸 + 几处指向已删文档的注释）。历史结论里**已被推翻**的三条
> （"6P 只留两三个"、"放弃混合引擎"、"cos⁴ 直积固有差异"）在汇编的"现状"行里
> 逐条点名——照旧文档施工前先看那里。
>
> 自检这一轮的**红线自查**：没有为压时间删掉任何一条断言（退掉的只有随功能消失的
> `gui_basis_presets_test.py`）；五个重体检**移到按需档而非删除**；30 个脚本的
> 子条件数与原断言数**逐节相等**；并发档 flaky 是改**测试的时序假设**（事件驱动），
> 不是放宽断言。六条"故意弄坏 → 必须红"的致死性证明见提交 `checks: 自检重设计…`。
