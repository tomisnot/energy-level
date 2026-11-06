# Energy Level

Rb-87 里德堡原子的能级、驱动演化和多原子仿真软件，提供桌面 GUI 与供 AI 操作者调用的语义接口。

软件负责物理模型、数值计算、配置和运行记录；人和 AI 负责选择实验、探索参数和解释结果。GUI 与 AI 接口通过 mecha 接入写权与留痕机制。

> **新接手的 AI 或人请先读 [接手必读](docs/AI_HANDOFF.md)**：进程/端口铁律、配置与卡片、多体模型货币、MPS 四条硬规则、症状对照表与已验证基准值都在那里。

## 主要能力

| 能力 | 内容 |
|---|---|
| 静态原子 | 能级组、AC/DC Stark、偶极矩阵元与谱线分析 |
| 单体与两体演化 | sambe、hard、mixed 等引擎，以及 RWA 参照；具体适用范围见引擎文档 |
| 多原子体系 | 阵列建模、相互作用、ED 等求解路径与观测量 |
| 参数探索 | 计算成本估计、预览、配置变化量、组合调用和独立扫描 runner |
| 交互与记录 | PyQt5 GUI、MCP 工具、后台任务、写权控制和只读监控 |

选择引擎时应结合近似条件和计算规模。数值结果、算法边界和物理含义见 [用户手册](docs/用户手册.md) 与 [引擎能力边界](docs/引擎能力边界.md)。

## 安装

使用 Python 3.10+；桌面模式需要可用的 Qt 图形环境。数值依赖和版本在 `requirements.txt` 中声明。

```bash
git clone https://github.com/tomisnot/mecha.git
git clone https://github.com/tomisnot/energy-level.git
cd energy-level
python -m venv .venv
```

激活虚拟环境：Windows PowerShell 使用 `.venv/Scripts/Activate.ps1`，Linux/macOS 使用 `source .venv/bin/activate`。随后在项目根执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e "../mecha[mcp]"
```

依赖清单引用 mecha 的 `v0.2.0` 发布点；上面的最后一步将框架切到同时检出的源码版本，用于当前开发代码。目录布局不同，请替换 `../mecha`。

mecha 是本软件的运行依赖，GUI 与 AI 接口都会使用它。静态能级依赖 ARC；遇到数据目录问题可运行 `python scripts/setup_arcdata.py` 查看准备步骤。

## 启动

```bash
# 本地桌面界面
python app_entry.py --gui

# 本软件的 MCP 服务与只读监控
python app_entry.py --mcp

# 在同一个软件进程中提供 MCP 并打开 GUI
python app_entry.py --mcp --with-gui

# 整站：由 mecha 编排软件与 AI 宿主
python -m mecha --all --root .
```

### AI 运行卡片

AI 服务启动时会尝试打开本机只读运行卡片，端口保存在 `.run-monitor-port`；设置
`EL_OPEN_RUN_MONITOR=0` 可关闭自动打开。AI 的 `ElEngine.run` 与 GUI 的
`ElOps.prepare/execute` 共用卡片存储，每次调用保留真实配置、进度、结果及失败原因。
多体结果优先投影到原子阵列：节点占据、自旋箭头、关联连线、闭环和纠缠子系统；
所有动画统一 8 秒一轮，曲线保留真实物理坐标。具体覆盖和边界见
[运行卡片说明](web/run_monitor/README.md)。

单独查看小规模真实计算的效果可运行：

```bash
python scripts/preview_run_monitor.py
```

命令输出本机动态地址，依次计算六原子的基态、有限温、态密度与演化；产物放在
`.mecha-v2/visual-preview`，不混入正式运行记录。单独静态托管网页只显示等待连接，
不会伪造结果。多体核心的数据契约与后端能力边界见
[多体接口与维护约定](docs/多体接口与维护约定.md)。

独立 `--gui` 模式不启动网络端点。需要人机操作同一个运行实例时，使用 `--mcp --with-gui` 或统一启动器提供的手动面。

AI 服务与监控使用动态端口，通过项目根的发现文件供客户端读取。写权与进程存活独立：切换人/AI 写权不需要重启软件，也不需要删除运行记录。

整站 AI 界面需要 dsh 可执行命令和本仓的 `dsh/` 插件资产。插件通过本地 `file:` 引用共享面板；换目录布局时，请先将 `dsh/package.json` 的 `@mecha/dsh-panel` 指向所用 mecha 仓的 `mecha/dsh-panel/`，再在 `dsh/` 执行 `npm install` 和 `npm run bundle`。

## AI 使用方式

AI 经工具面读配置、查能级、预览模型、估计成本、发起演化和读取结果。工具参数、配置键与单位见 [操作指南](skills/energy-level/SKILL.md)。

- 配置写入经受控入口，留有操作理由与前后状态。
- `run_actions` 支持组合调用、`spec_patch` 变化量和 `pick` 结果选择，便于参数探索。
- `submit_job` / `read_job` / `cancel_job` 用于后台演化；取消是协作式请求，实际终态以任务查询为准。
- AI 组合调用中的扫描目前按步骤串行执行；独立扫描 runner 的并行能力不等于工具面已有并行扫描入口。
- 物理索引与配置值应从查询取得，失败按结构化提示纠正。

mecha 提供连接与治理机制；原子模型、引擎适用条件、参数单位和物理判断属于本项目。

## 开发与验证

```bash
python run_checks.py                 # 主干 smoke（默认，<1 min）
python run_checks.py --cascade       # smoke 通过即停；失败展开 core
python run_checks.py --changed       # 按工作区改动选判据
python run_checks.py --core          # 核心契约与小尺寸物理结果
python run_checks.py --fast          # 提交前较宽场景
python run_checks.py --all           # 常规全量
python run_checks.py --only sambe    # 单项
```

默认只跑主干：一坏就必坏全局的契约与最小物理锚。`--cascade` 用“少量实验判断有没有问题，有问题再展开”；`--changed` 把改动路径映射到判据。`--all` 运行常规全量，`--slow` 加入慢物理基准，`--release` 单独进行发布扫描。可用 `--list` 查看范围，详见 [验证说明](checks/README.md)。

验证保护数值结果、引擎契约和实际接线。按改动范围选择测试，不要求每次开发执行慢基准，也不以文档措辞、历史文件名或额外红证作为开发门禁。

修改 dsh 接线时，在 `dsh/` 执行 `npm test`，检查 TypeScript 和浏览器构建。

## 代码与文档

```text
src/core/       物理模型、参数声明、数值引擎与统一运行分派
src/ui/         桌面界面
src/sweep/      独立扫描 runner
mecha_v2/       领域引擎适配、工具、GUI 桥与 MCP 服务
src/v2_bind.py  软件装配与服务接线
dsh/            AI 宿主中的项目面板
skills/         AI 操作指南
checks/         结构、数值结果与集成验证
docs/           物理、使用与设计文档
```

物理层与框架分开；`mecha_v2/` 将本项目的能力接到 [mecha](https://github.com/tomisnot/mecha)。同一框架也用于 [ML Toolbox](https://github.com/tomisnot/ml-toolbox)、[PaperPilot](https://github.com/tomisnot/get-paper) 和 Tutor，各项目独立维护自己的领域内容。

- [用户手册](docs/用户手册.md)
- [开发者指南](docs/开发者指南.md)
- [引擎能力边界](docs/引擎能力边界.md)
- [多体接口与维护约定](docs/多体接口与维护约定.md)
- [多体开发纲领](docs/多体开发纲领.md)
- [多体观测量与共享内核施工规划](docs/施工规划-多体观测量与共享内核.md)
- [有限 MPS 分析与资源契约](docs/有限MPS分析与资源契约.md)
- [扇区扫描与精确基线](docs/扇区扫描与精确基线.md)
- [代码总览](docs/代码总览.md)
- [历史文档](docs/history/)

## 当前边界

数值方法的近似条件与成本各不相同，大基组和多原子体系需要先评估规模。后台任务表保存在当前进程中；进程重启后的追踪与恢复不应仅依赖 `job_id`。本地 GUI、真实协议与数值标杆分别验证，具体覆盖和环境依赖见验证说明。

GUI 处于维护态：主工作流是 AI/MCP；静态/动态页保持，多体页暂不大改。多体页旧键接线尚未迁移，多体/检视器/扫描栏的深测已归档，默认只保留静态/动态页与最小装配判据。

## 有限 MPS 静态计算与诊断

通用 DMRG 支持安全实数张量、原生基态接续、显式站点顺序、受预算约束的精确方差与块置换期望；Python API 还支持显式二阶置换扇区准备与投影权重、压缩泄漏报告。配置和能力边界见 [有限 MPS 分析与资源契约](docs/有限MPS分析与资源契约.md)。

## 许可

[MIT](LICENSE)。
