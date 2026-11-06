# 验证的运行方式

架构先消除错误空间，回归验证真实结构与结果。不以文档格式、工具总数、历史文件或词汇黑名单代替设计审查，不要求每条回归配变异红证。

```powershell
python run_checks.py                   # 默认：主干 smoke（<1 min）
python run_checks.py --cascade         # smoke 通过即停；失败展开 core
python run_checks.py --changed         # 按工作区改动路径选判据
python run_checks.py --core            # 核心契约与小尺寸物理结果
python run_checks.py --only sambe      # 按名称选择，不受默认档位限制
python run_checks.py --fast            # 较宽的快速场景
python run_checks.py --all             # 常规全量：GUI、真实 MCP、物理标杆
python run_checks.py --slow            # 再加第一性原理慢基准
python run_checks.py --release         # 发布前个人路径扫描
python run_checks.py --all --list      # 只查看范围
```

主干只放“一坏就必坏全局”的契约与最小物理锚；具体模块改动由 `--changed` 展开，
失败后再用 `--cascade` 或 `--core` 定位。每项独立进程，单项超时与失败输出可见。

| 档位 | 规模 | 用途 |
|---|---|---|
| 默认 `--smoke` | 11 项 / ~67 s | 每次改动的第一道门 |
| `--cascade` | smoke → 失败才 core | 少量实验判断，有问题再分叉 |
| `--changed` | 按 git 改动映射 | 只跑相关模块（无映射退回 smoke） |
| `--core` | 44 项 / ~4 min | 共享层改动、合并前 |
| `--fast` | 更宽场景 | 提交前 |
| `--all` | 全部常规判据 | 定期/发布前 |
| `--slow` | 加第一性原理慢基准 | 单独排期 |

## GUI 判据

GUI 已进入维护态，主工作流是 AI/MCP。默认只保留静态/动态页与最小装配的判据：
`gui_app_entry`、`gui_bind`、`gui_run_flow`、`gui_run_worker`、`levels_twopage`、
`inprocess_gui`。多体页、检视器、扫描栏的深测已移到 `checks/archive/gui/`，
默认不跑；恢复相应界面开发时移回并登记档位即可。

面板接线变动时在 dsh 运行 npm test（TypeScript + 现有 platform=browser 真实构建）。共享面板行为归框架；项目侧不另维护源码 import 解析器、固定 bootstrap 拼写或文件黑名单。

退役：cleanliness_guard（迁移符号黑名单）、ai_surface（技能登记/措辞/工具表镜像）、capability_map（生成文档新鲜度）；checks_sentinel 只保留真实的兼容包导出回归。能力快照仍可用 scripts/gen_capability_map.py 按需刷新。技能文档说明常用工作流，完整工具 schema 来自宿主注册表。

本次验证：核心 18 项通过（38s）；常规全量 61 项已执行，控制链路在全量初跑出现进程/协议失败，原版与整改版独立复跑均通过。保留该真实链路用例，未改为跳过。物理标杆中的一个可选分支报告 SKIP；第一性原理慢基准仅由 --slow 选择。面板类型与真实构建通过。
