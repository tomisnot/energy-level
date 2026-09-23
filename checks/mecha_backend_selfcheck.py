# -*- coding: utf-8 -*-
"""硬件档预留自检（Phase 5）——只验"结构就位"，不接真机。

  A 后端路由表：带 `backend` 列（harvest 的 DISPATCH 北极星）；'sim' 实现、'hardware' 预留。
  B backend='sim' 逐位等价现状 + 默认 + **AI 侧零改动**（工具面仍是那 7 个，没为硬件改 API）。
  C L3 双闸（铁律二）：sim auto_pass；hardware 需 人确认 + 硬件互锁 + 已武装 **三闸齐过**，
    缺一即拒（且默认 ARMED=False ⇒ AI 提议也放行不了）。
  D hardware 预留未实现：route/Ops 明确报 ReservedBackend（不静默退化成 sim）。
  E 数字孪生 dry-run（§6 统一点）：propose 先在 sim 跑预测，armed=False、requires_confirmation，
    发一条 propose 事件（AI 只能提议）。
  F 分级自治 classify：只读→L1、可逆写→L2、sim 运行→L2、hardware 运行→L3、未知→L3（最严）。

真跑最小演化（单体 mixed，6 态，10 ns）验证 sim 路径。用临时 root。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from arc import Rubidium87                                 # noqa: E402
from src.core.static_atom import ArcCache                  # noqa: E402
from src.core import runspec as RS                         # noqa: E402
from src.core import waveforms as WF                       # noqa: E402
from src.core.atomic_levels import state_key               # noqa: E402
from mecha.session import Session                      # noqa: E402
from mecha.ops import Ops                             # noqa: E402
from mecha import autonomy as AU                       # noqa: E402
from mecha.autonomy import ReservedBackend             # noqa: E402
from mecha import server as SVR                        # noqa: E402
from mecha import runtime                              # noqa: E402
from src.adapter import EngineAdapter                      # noqa: E402

runtime.set_engine(EngineAdapter())   # 注入领域引擎（mecha 零领域 import）

C = Checker('mecha_backend')

# ---------------- A. 后端路由表（backend 列） ----------------
bt = AU.backends()
C.group('A 后端路由表：带 backend 列（sim 实现 / hardware 预留）', [
    ("路由表含 'sim' 与 'hardware' 两个 backend",
     'sim' in bt and 'hardware' in bt),
    ("sim：implemented=True、reversible=True、l3_gate=auto_pass",
     bt['sim']['implemented'] is True and bt['sim']['reversible'] is True
     and bt['sim']['l3_gate'] == 'auto_pass'),
    ("hardware：implemented=False、reversible=False、l3_gate=propose_confirm_interlock",
     bt['hardware']['implemented'] is False
     and bt['hardware']['reversible'] is False
     and bt['hardware']['l3_gate'] == 'propose_confirm_interlock'),
    ('每个 backend 都带 label（可读展示）',
     all('label' in v for v in bt.values())),
])

# ---------------- 建最小可运行配置（供 B/E 的 sim 路径） ----------------
_TMP = tempfile.mkdtemp(prefix='mecha_p5_')
_cache = ArcCache(Rubidium87())
_GROUPS = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=1.5)]
_st = RS.build_states(_cache, _GROUPS, 0.0, 0.0)
_freq = abs(_cache.E_mhz(60, 1, 1.5) - _cache.E_mhz(60, 0, 0.5))
_wave = WF.Wave('opt', freq_mhz=_freq,
                envelope=WF.SquareEnv(0.0, 1.0, 1.0, 8.0), pol='π',
                intensity_mwcm2=1.0)
_CFG = {'groups.list': _GROUPS, 'field.B_gauss': 0.0, 'field.E_vpcm': 0.0,
        'dynamic.mode': 0, 'dynamic.method': 'mixed', 'dynamic.t_max': 10.0,
        'dynamic.npts': 11, 'dynamic.decay_index': 1,
        'dynamic.psi_single': state_key(_st[0]),
        'dynamic.waves_single': WF.serialize_waves([_wave])}
_sess = Session(root=_TMP)
for _k, _v in _CFG.items():
    _sess.set(_k, _v, actor='setup', reason='判据初始化')

# ---------------- B. backend='sim' 等价现状 + AI 侧零改动 ----------------
_ops_default = Ops(_sess)                 # 不传 backend → 默认 sim
_ops_sim = Ops(_sess, backend='sim')
r_sim = _ops_sim.run(actor='ai', reason='sim 路径验证')
r_def = _ops_default.run(actor='ai', reason='默认路径验证', use_cache=True)
C.group("B backend='sim' 逐位等价现状 + 默认 + AI 侧零改动", [
    ("Ops() 默认 backend='sim'", _ops_default.backend == 'sim'),
    ("route('sim') 返回已实现路由", AU.route('sim')['implemented'] is True),
    ('sim run 成功（走 runspec 唯一分派，I5）',
     r_sim['ok'] is True and r_sim['backend'] == 'sim'),
    ('默认与显式 sim 同配置→同 artifact_id（逐位等价）',
     r_def.get('artifact_id') == r_sim.get('artifact_id')),
    ('**AI 侧零改动**：backend 是 ops 的路由细节，硬件档不新增 MCP 工具（工具面 backend-agnostic）',
     # 17 = P1-9d 后的**通用**工具数（领域工具 set_drive/get_waves 已搬回领域侧）
     # + P1-6 的泛化 query（1 个通用工具，后端无关）
     len(SVR.TOOL_NAMES) == 17
     and not any('hardware' in t for t in SVR.TOOL_NAMES)),
])

# ---------------- C. L3 双闸（铁律二） ----------------
g_sim = AU.l3_gate('sim')
g_hw_none = AU.l3_gate('hardware')
g_hw_conf = AU.l3_gate('hardware', confirmed=True)
g_hw_ci = AU.l3_gate('hardware', confirmed=True, interlock_ok=True)   # 仍未武装
g_hw_all = AU.l3_gate('hardware', confirmed=True, interlock_ok=True, armed=True)
C.group('C L3 双闸：sim 放行；hardware 需 人确认+互锁+武装 三闸齐过', [
    ('sim 档 auto_pass（可逆可重跑，L3 默认放行）', g_sim[0] is True),
    ('hardware 默认拒（未武装/未确认/未互锁）', g_hw_none[0] is False),
    ('仅人确认不够（还缺互锁+武装）→ 拒', g_hw_conf[0] is False),
    ('人确认+互锁但**未武装**（ARMED=False）→ 仍拒（AI 提议也放行不了）',
     g_hw_ci[0] is False),
    ('三闸齐过（确认+互锁+武装）→ 放行', g_hw_all[0] is True),
    ('模块级 ARMED 默认 False（安全默认）', AU.ARMED is False),
])

# ---------------- D. hardware 预留未实现 ----------------
try:
    AU.route('hardware'); route_raised = False
except ReservedBackend:
    route_raised = True
try:
    Ops(_sess, backend='hardware'); ops_raised = False
except ReservedBackend:
    ops_raised = True
try:
    AU.route('quantum-sim'); unknown_raised = False
except ReservedBackend:
    unknown_raised = True
C.group('D hardware 预留未实现：明确报 Reserved（不静默退化成 sim）', [
    ("route('hardware') → ReservedBackend", route_raised),
    ("Ops(backend='hardware') → ReservedBackend", ops_raised),
    ("route(未知 backend) → ReservedBackend", unknown_raised),
    ('ReservedBackend 是可教学错误（带 kind/hint）',
     issubclass(ReservedBackend, Exception)
     and ReservedBackend.kind == 'reserved_backend'),
])

# ---------------- E. 数字孪生 dry-run（§6 统一点；附告二 后**域参数化**） ----------------
# 旧版库内写死 `l3_gate('hardware')` / `target='hardware'`；现在目标由**领域**给。
# 三种情形都验：未指定目标（最严、不放行）、显式 hardware、领域登记的不逆 op。
seq0 = _sess.bus.last_seq
prop = _ops_default.propose(reason='提议一段序列（先 sim 预演）', actor='ai')
prop_events = [e for e in _sess.activity(seq0) if e.get('kind') == 'propose']
_prop_hw = _ops_default.propose(reason='显式指定 hardware 目标',
                                actor='ai', backend_target='hardware')
AU.register_irreversible_ops({'fire_laser': '出光——不可逆、安全攸关'})
_prop_op = _ops_default.propose(reason='领域登记的不逆 op', actor='ai',
                                op='fire_laser')
AU.register_irreversible_ops({})          # 还原，避免影响后续
C.group('E 数字孪生 dry-run：AI 提议→先 sim 跑→摆给人（不武装）＋目标**域参数化**', [
    # dry-run 摘要 = estimate + **整份 preview 原样搬运**（P1-9c：库不挑领域键。
    # 旧版在这里拼 n_states/basis_enough/driven 三个键——那正是"库替领域决定
    # 哪几个键重要"的残留；非物理引擎的 preview 里根本没有这些键。）
    ('propose 跑了 dry-run（estimate + preview 原样透传，基组规模 6）',
     'estimate' in prop['dry_run_sim']
     and (prop['dry_run_sim'].get('preview') or {}).get('n_states') == 6),
    ('**未指定目标 ⇒ 按最严**：不放行（旧版靠写死 hardware 才恒拒）',
     prop['proposal']['backend_target'] is None
     and prop['l3_gate']['allow'] is False),
    ('**显式 hardware 目标**：仍不放行（armed=False）且 armed 标 False',
     _prop_hw['proposal']['backend_target'] == 'hardware'
     and _prop_hw['armed'] is False and _prop_hw['l3_gate']['allow'] is False),
    ('**领域登记的不逆 op**（未指定 backend）→ 走不可逆档、不放行',
     _prop_op['l3_gate']['allow'] is False
     and '不可逆 op' in str(_prop_op['l3_gate']['reason'])),
    ('requires_confirmation=True（单向门必须人把关）',
     prop['requires_confirmation'] is True),
    ('发了一条 propose 事件（M1 归因，进总线）', len(prop_events) >= 1),
    ('`is_irreversible` 只认登记过的 op（未登记 → 假）',
     AU.is_irreversible('fire_laser') is False
     and AU.irreversible_ops() == {}),
])

# ---------------- F. 分级自治 classify ----------------
C.group('F 分级自治 classify（统一判据：可逆性/代价）', [
    ("只读 → L1（get_state/preview）",
     AU.classify('get_state') == 'L1' and AU.classify('preview') == 'L1'),
    ("可逆写 → L2（set_param/undo）",
     AU.classify('set_param') == 'L2' and AU.classify('undo') == 'L2'),
    ("sim 运行 → L2（可逆可重跑）",
     AU.classify('run_compute', 'sim') == 'L2'),
    ("hardware 运行 → **L3**（不可逆）",
     AU.classify('run_compute', 'hardware') == 'L3'),
    ("未知操作 → L3（最严安全默认）",
     AU.classify('self_destruct') == 'L3'),
])

sys.exit(0 if C.summary() else 1)
