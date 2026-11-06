# 归档判据

这些判据默认不跑；它们保护的是**暂时退役/低优先级**的界面面，不是产品契约。
恢复对应界面开发时，把文件移回 `checks/` 并登记到 `run_checks.py` 的合适档位即可。

## gui/

| 文件 | 保护什么 | 归档原因 |
|---|---|---|
| `gui_manybody_selfcheck.py` | 多体页装配与旧 `manybody.atoms` 接线 | 多体页暂停大改；旧键已退役，当前预期红 |
| `gui_inspector_selfcheck.py` | 上帝模式检视器旋钮/接线 | GUI 进入维护态，检视器只服务 mixed diag |
| `sweep_gui_selfcheck.py` | 底栏扫描栏的形态/收放/取消 | 扫描 UI 非 AI/MCP 主路径 |

保留在 `checks/` 的 GUI 判据只守静态/动态页与最小装配：

- `gui_app_entry_selfcheck.py`：MainWindow 能起来；
- `gui_bind_selfcheck.py`：控件↔Gate 绑定与写权门控；
- `gui_run_flow_selfcheck.py`：运行流走 `ops`；
- `gui_run_worker_selfcheck.py`：主线程 prepare / 工作线程 execute；
- `levels_twopage_selfcheck.py`：静态页与动态页能级基准一致；
- `inprocess_gui_selfcheck.py`：`--mcp --with-gui` 同桥与控制链路。

不跑归档判据不等于删除覆盖：相关契约（Gate、线程、控制端点、calib）仍由
`manybody_*`、`control_channel`、`authority`、`ops_threadsplit` 等判据保护。
