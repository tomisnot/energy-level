<!-- 本文件由 scripts/gen_capability_map.py 从角色声明生成，禁止手工编辑。
     改能力归属请改 mecha/assembly.py 的 declare_roles，然后重新生成。
     字段格式固定，供第二个项目直接并表（总纲 §5.4 交付 2/3）。 -->
# Capability Map（生成）

| 能力 | 角色 | provider | 消费方 |
|---|---|---|---|
| `adapter.ai` | consumer | energy-level | ai |
| `adapter.gui` | consumer | energy-level | human |
| `authority` | definition | energy-level | — |
| `commands.registry` | consumer | energy-level | adapter.ai |
| `engine` | definition | — | — |
| `engine` | provider | ElEngine | — |
| `gate` | definition | energy-level | — |
| `history` | definition | energy-level | — |
| `monitor` | definition | energy-level | — |
| `monitor.transport` | definition | energy-level | — |
| `monitor.transport.local` | provider | http-loopback | — |
| `ops_pipeline` | consumer | — | Surface |
| `provider.approval` | definition | energy-level | — |
| `provider.approval.local` | provider | fail-closed | — |
| `provider.artifacts` | definition | energy-level | — |
| `provider.artifacts.local` | provider | local-file | — |
| `provider.codeact` | provider | actions-interp | adapter.ai |
| `provider.jobs` | definition | energy-level | — |
| `provider.jobs.local` | provider | thread | tools.registry |
| `provider.log` | definition | energy-level | — |
| `provider.log.local` | provider | jsonl | history |
| `provider.skills` | provider | catalog | — |
| `provider.spill` | definition | energy-level | — |
| `provider.spill.local` | provider | local-file | — |
| `provider.toolhost` | definition | energy-level | — |
| `provider.toolhost.local` | provider | in-process | — |
| `provider.toolhost.mcp` | provider | mcp-streamable-http | — |
| `query:preview_run` | consumer | preview_run | framework:CodeAct.run_actions |
| `query:query_levels` | consumer | query_levels | framework:CodeAct.run_actions |
| `query:query_params` | consumer | query_params | framework:CodeAct.run_actions |
| `query:read_config` | consumer | read_config | framework:CodeAct.run_actions |
| `query:read_waves` | consumer | read_waves | framework:CodeAct.run_actions |
| `surface` | definition | energy-level | — |
| `surface.engine` | provider | ElEngine | surface |
| `tools.registry` | consumer | energy-level | adapter.ai |
| `visibility.policy` | definition | energy-level | — |
