"""EL 的 mecha v2 接入层（原 mecha 仓 examples/energy_level，2026-09-24 搬回）。

**个性知识归所属项目**（ADR `领域参考实现搬回所属项目`）：EL 的领域实现
（物理引擎 Provider、AI 工具清单、GUI 适配器、独立 Hub）住在这里，不寄存在
通用框架仓。对框架只依赖 `mecha.*` 公开面；这里允许出现 EL 领域值
（B=Gauss、Rb-87 等）——框架反模式禁的是**库里**写死项目值。

模块地图：
- `engine.py`  ElEngine（Surface 的 Provider：schema/run/summary/estimate）+
  make_validator（委托 src.core.protocol 的领域校验）
- `tools.py`   build_tool_registry——AI 的工具面（set_config/run_sim/set_drive…）
- `bridge.py`  V2Bridge——GUI ↔ 同一道 Gate 的操作者适配器（+ ElOps 线程拆分）
- `hub.py`     to_mcp——ToolRegistry → MCP 服务器的通用投影 + 独立进程入口
  （`python -m mecha_v2.hub`，headless 部署）

**只 import `src.core.protocol`**（EL 的领域无关、框架无关纯领域层，docstring
明写"本协议不 import mecha"），绝不碰 v1 的 `src.adapter`/`mecha.app_bridge`
——那是旧形状，读了会锚定（用户纪律 + §11）。

## Known Limitations and Deferred Work

- cache 在构造时建一次（ARC 首次 ~13 s）并进程级复用；多原子种类未参数化
  （EL 当前只 Rubidium87，第二个原子出现再抽）。
- 取消经后台线程把 `progress_arr[1]` 置 1（EL 的取消约定）；引擎不响应取消时
  退化为跑完，不硬杀（同进程无法强杀，与 dsh 工具取消语义一致）。
"""
