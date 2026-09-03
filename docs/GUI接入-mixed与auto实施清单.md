# GUI 接入实施清单：method='mixed' + 'auto'（目标 5）

> 2026-09-06 船长勘察固化。依赖：t62 混合引擎验收完成（8 用例 + numba RHS +
> 22 态两体 dim 1452）。本文档是 t62 完成后的**照单执行清单**（行号已核实）。
> 状态：勘察完成，待 t62 验收后执行。

---

## 0. 前置条件（执行前必须满足）

1. t62 全绿：8 用例 + numba RHS（用例 4 STIRAP 链不再 13 分钟）+ 22 态两体 620ns
   dim 1452 验收通过。
2. evolve_mixed 最终 API 确定：`evolve_mixed(states, opt_meta, psi0, t, ...)`
   单原子入口签名 + 两体入口（若有）——以 t62 提交为准。
3. route_auto 两体+微波分支语义定稿：mixed 两体可用后，"强光+微波+两体"
   应从 sambe 改为 mixed（5 倍降维 dim 1452 vs 7260）——改 route.py L88-89。

## 1. method_cb 档位（src/ui/ui_dynamic.py L224-232）

- 选项：`['ip4', 'sambe', 'hard', 'mixed', 'auto']`（mixed 在 hard 后、auto 最后）
- 默认：维持 'hard'（t25 决策）——auto 不设默认，避免行为突变
- L231 `_on_method_changed`：加档位说明（tooltip/状态栏），auto 档可提示
  "按场与能级智能路由（route_auto）"

## 2. 单原子演化路径（src/ui/ui_dynamic.py `_run_single_body` L895-942）

method 分支现状：
- 'sambe' L899-902 → `build_fields_exact`
- 'hard' L903-917 → `build_two_field` + `arbitrary_mw_group`（ArbitraryWave 并入）
- else（'ip4'）L918-923 → `build_fields`

**auto 分支**（加在 method 解析最前，L898 后）：
```python
if method == 'auto':
    from ..core.route import route_auto
    m, reason = route_auto(states, waves, two_body=False, cache=self.cache)
    self.main.log(f'auto 路由 → {m}: {reason}')
    method = m          # 落到下面具体引擎分支
```
- route_auto 用 cache=self.cache（主线程 ArcCache，精确失谐判据）
- ArbitraryWave 场景 route_auto 必回 'hard' → 走 L903-917 分支天然正确

**mixed 分支**（加在 'hard' 分支后）：
```python
elif method == 'mixed':
    # 光场组（Sambe 光子块）+ 微波当场——t62 evolve_mixed 入口
    from ..core.evolve_mixed import evolve_mixed   # 以 t62 最终 API 为准
    fields = build_mixed_meta(self.cache, states, waves)  # 待 t62 提供构造器
```
- mixed fields 形态与构建器以 t62 提交为准（t62 前 GUI 只有原型 tempsimu/_t62_hybrid.py）

## 3. 两体演化路径（src/ui/ui_dynamic.py `_run_two_body` L944+）

- method 解析同单原子（L949）；A/B 波形独立
- auto：对 A、B 分别 route_auto（two_body=True）→ 档位不一致取精度保守档
  （sambe/mixed > hard；mixed 两体可用前保守回 sambe——同 route.py L88 现状）
- mixed 两体：EvolveWorkerPair 加 'mixed' 分支（用例 8 V_dd φ-帧相位验收后）
- V_dd 预计算与 per_freq_M/frozen_h_ns 参数传递同现有 'sambe' 分支（L969-974）

## 4. Worker 分支（src/ui/evolve_worker.py）

单原子 `EvolveWorker.run` L32-57：
- L34 'hard' 分支不动
- 加 `if self.method == 'mixed':` → `evolve_mixed(...)`（progress_arr/log/decays 透传，
  结构同 hard 分支 L38-46）
- else（'sambe'/'ip4'）不动
- 'auto' **不应到达 worker**（UI 层已解析成具体档）——若到达按未知档处理报错

两体 `EvolveWorkerPair.run` L105-145：
- 加 'mixed' 分支（t62 两体入口），hard/exact/ip4 不动

## 5. 序列化兼容（src/ui/ui_dynamic.py L620-691）

- save L632 `'method': self.method_cb.currentText()`——'mixed'/'auto' 直接存字符串，无需改
- restore L686-691：
  ```python
  _m = {'auto': 'ip4', 'exact': 'sambe'}.get(...)   # 旧档 'auto' = 老 RWA 后端，映射保留
  if _m not in ('ip4', 'sambe', 'hard', 'mixed', 'auto'):
      _m = 'hard'
  ```
  → 白名单加 'mixed'/'auto'
- 注意：restore 里旧档映射 `'auto'→'ip4'` 保留（语义 = 老后端选择，非新智能路由）

## 6. 回归

1. `checks/ui_selftest.py`（保存/恢复 + 各档演化 smoke）
2. route_auto 自检 6/6（checks/route_auto_selfcheck.py）
3. 新增：auto 档端到端 smoke（构造 STIRAP/微波/弱光场景 → 确认路由目标引擎
   真的被调用——worker 打日志）
4. 两体 auto：A/B 混合档位场景 1 个

## 7. 评审点（执行时向用户确认）

1. 'ip4' 是否保留在列表（退役但兼容旧档对比）？建议保留
2. auto 显示 reason 的位置：状态栏 log（建议）vs method_cb tooltip
3. 列表顺序/默认值（当前建议 hard 默认，auto 最后）
