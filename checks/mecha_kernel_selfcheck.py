# -*- coding: utf-8 -*-
"""机甲语义内核自检（Phase 1）——registry / bus / session 的不变式 + 致死性证明。

守的是纲领的硬约束在 Re0 里真的成立：

  A. **收集不是复制**：registry 扫出 core 的 32 个 `Param`，值逐位不变（C1 单一定义点）。
  B. **渐进披露**：list_params 默认藏 internal（D-B）；未知键给 did_you_mean（D2）。
  C. **I6 契约外字段永远 None**：配置 schema ⊇ runspec 实读键（AST 交叉，防漂）。
  D. **M1/M2 归因 + 声称vs实测**：每次 set 恰好一条带 actor/reason/before/after 的
     事件；回执 readback 由 choke point 产出、== resolved（AI 伪造不了右列）。
  E. **D5 报错不静默**：非法值被拒且快照不变（自愈回退）；未知键/空 reason 报错。
  F. **可逆**：undo 恢复 before 并发反向事件。
  G. **I3/I4/M4**：大载荷走指针+sha1；体积闸截断必告知；append-only 重启续 seq。
  H. **I1 无侧门（致死性）**：AST 证明改 `_snapshot` 只在 session 的 __init__/_write；
     并**故意喂一段裸写入**证明扫描器真会亮红（不是永不触发的哑守卫）。

用**临时目录**当 bus root（不污染 `<repo>/.mecha`）。
"""
import ast
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                       # noqa: E402
from src.mecha import registry as R                       # noqa: E402
from src.mecha import session as SESS                     # noqa: E402
from src.mecha.bus import Bus                             # noqa: E402
from src.mecha.errors import InvalidValue, UnknownKey, MechaError  # noqa: E402
from src.core import runspec as RS                        # noqa: E402
from src.core import protocol as CP                       # noqa: E402
from src.mecha import runtime                             # noqa: E402
from src.adapter import EngineAdapter                     # noqa: E402

runtime.set_engine(EngineAdapter())   # 注入领域引擎（mecha 零领域 import）
C = Checker('mecha_kernel')
_MECHA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'src', 'mecha')

# 32 个引擎参数的 golden 值（逐位对拍：包 Param 后引擎读到的值不变）
GOLDEN = {
    'PHASE_TOL': 1e-4, 'R_CRIT': 0.0135, '_RHS_ENV_TAB_N': 20001, 'PHASE_BUDGET': None,
    'DEFAULT_RTOL': 1e-8, 'ATOL_RTOL_RATIO': 1e-2, 'DEFAULT_ATOL': 1e-10,
    'M_EST_TOL': 1e-6, 'BW_OPT': 2e4, '_AC_ENV_TAB_N': 4001,
    'M_EST_BW_MHZ': 2e4, '_M_START': 2, '_M_MAX': 8, '_DENSE_MAX': 2200,
    '_ENV_PROBE_N': 2001, '_DT_MIN_FRAC': 2e5, '_SPARSE_SEG_MAX': 50000,
    '_SPARSE_MAX_DEPTH': 4, '_NODE_LADDER': (16, 32, 64, 128), '_VOLT_MAX_ITER': 60,
    '_VOLT_PICARD_FRAC': 0.05, '_VOLT_PRE_SWEEPS': 8,
    'OPT_NRES_BW_MHZ': 2e3, 'FRAME_TOL_REL': 1e-6, '_TF_DENSE_MAX': 3000,
    'RWA_TOL': 1e-4, 'MW_MAX_MHZ': 1e5, 'RYDBERG_N_MIN': 20,
    'DIRECT_SOS_N_WINDOW': 3, 'AC_EFFECT_THRESHOLD_MHZ': 0.05,
    '_ENV_SAMPLES': 64, '_TAB_N': 2 ** 17,
    # manybody（第 7 引擎家族，并入 core 后同被 protocol._scan_param_defs 扫出）
    '_MB_DENSE_MAX': 4096, '_MB_FULLPOP_MAX': 65536, '_MB_MAX_DIM': 2 ** 20,
    'MB_N_WINDOW': 3, 'MB_GUARD_MHZ': 1.0, 'MB_ED_MAX_DIM': 2 ** 20,
}

# ---------------- A. 收集不是复制（C1 单一定义点） ----------------
ep = R.engine_params()
val_bad = []
for name, want in GOLDEN.items():
    rec = ep.get(name)
    if rec is None:
        val_bad.append('%s(缺)' % name)
        continue
    got = rec['default']
    if not (got == want):
        val_bad.append('%s(%r!=%r)' % (name, got, want))
tiers = {}
for rec in ep.values():
    tiers[rec['tier']] = tiers.get(rec['tier'], 0) + 1
# solve.* 键从 SOLVE_KWARG 派生（不是手抄）
solve_keys = set(R.solve_schema())
derived_ok = solve_keys == {'solve.' + k for k in RS.SOLVE_KWARG}
C.group('A registry 收集 core 的 Param（值逐位不变 + solve.* 派生）', [
    ('收集到 %d 个引擎参数（= golden 数 %d）' % (len(ep), len(GOLDEN)),
     len(ep) == len(GOLDEN)),
    ('%d 个值逐位等于 golden（不符：%s）' % (len(GOLDEN), val_bad or '无'),
     not val_bad),
    ('每个 record 带 name/home/type/unit/desc/default/tier',
     all({'name', 'home', 'type', 'desc', 'default', 'tier'} <= set(r)
         for r in ep.values())),
    ('tier 分层存在（internal 供深挖）：%s' % tiers, tiers.get('internal', 0) >= 15),
    ('solve.* 键从 runspec.SOLVE_KWARG 派生（不手抄）', derived_ok),
])

# ---------------- B. 渐进披露 + 可教学错误 ----------------
lp_def = R.list_params()
lp_all = R.list_params(include_internal=True)
hidden = lp_all['count'] - lp_def['count']
try:
    R.describe('field.B_gaus')            # 打错一个字母
    sugg = []
except UnknownKey as e:
    sugg = e.to_dict().get('suggest', [])
C.group('B list_params 分页/tier 过滤 + did_you_mean（D-B/D2）', [
    ('默认藏 internal：%d < 全量 %d（藏了 %d）'
     % (lp_def['count'], lp_all['count'], hidden), hidden >= 15),
    ('list_params 返回项带 name/family/type',
     all('name' in p and 'family' in p for p in lp_def['params'])),
    ('截断字段带 how_to_get_more（I4）',
     'how_to_get_more' in lp_def),
    ('打错键名给最可能的建议（field.B_gaus→%s）' % sugg,
     'field.B_gauss' in sugg),
])

# ---------------- C. I6：配置 schema ⊇ runspec 实读键 ----------------
lit, dyn = CP.runspec_read_keys()
fam = R.families()
lit_missing = [k for k in lit if k not in fam]
# 动态键家族（dynamic.waves_%s）须在 schema 里显式声明其成员
dyn_family_ok = all(k in R.config_schema()
                    for k in ('dynamic.waves_single', 'dynamic.waves_A',
                              'dynamic.waves_B'))
C.group('C 配置 schema 完全覆盖 runspec 实读键（I6 AST 交叉）', [
    ('runspec 字面读的键都在 schema（漏：%s）' % (lit_missing or '无'),
     not lit_missing),
    ('扫到动态键家族 %d 个（dynamic.waves_*）' % len(dyn), len(dyn) >= 1),
    ('动态键家族成员已显式声明（waves_single/A/B）', dyn_family_ok),
    ('schema 至少覆盖 groups/field/dynamic/solve 四族',
     len(R.config_schema()) >= 15 and len(R.solve_schema()) == 5),
])

# ---------------- D/E/F. session choke point（用临时 root） ----------------
tmp = tempfile.mkdtemp(prefix='mecha_chk_')
s = SESS.Session(root=tmp)
rec1 = s.set('field.B_gauss', 21.7, actor='ai', reason='Zeeman 调谐')
ev_after_1 = s.activity(0)
s.set('dynamic.t_max', 200, actor='ai', reason='加长窗口')
ev_after_2 = s.activity(0)
one_event_per_set = (len(ev_after_1) == 1 and len(ev_after_2) == 2)
e0 = ev_after_2[0]
has_attrib = all(k in e0 for k in ('actor', 'reason', 'before', 'after', 'seq',
                                   'kind', 'target', 'reversibility'))
# 非法值：拒 + 快照不变（D5）
snap_before = s.effective()
try:
    s.set('dynamic.t_max', -5.0, actor='ai', reason='坏值')
    rejected = False
except InvalidValue:
    rejected = True
snap_unchanged = (s.effective() == snap_before)
# 未知键 + 空 reason
try:
    s.set('dynamic.nope', 1, actor='ai', reason='x'); unk_ok = False
except UnknownKey:
    unk_ok = True
try:
    s.set('field.B_gauss', 1.0, actor='ai', reason=''); reason_ok = False
except MechaError:
    reason_ok = True
# undo（F）
und = s.undo(actor='ai')
und_ev = s.activity(0)[-1]
C.group('D set 是 choke point：每次恰好一条带归因的事件 + readback（M1/M2）', [
    ('2 次 set → 恰好 2 条事件', one_event_per_set),
    ('事件带 actor/reason/before/after/seq/kind/target/reversibility', has_attrib),
    ('e0.actor=ai、e0.reason 非空', e0['actor'] == 'ai' and bool(e0['reason'])),
    ('readback == resolved（choke point 产出的实测，M2）',
     rec1['readback'] == rec1['resolved'] == 21.7),
    ('before/after 记进事件（21.7 那次 before=None after=21.7）',
     e0['before'] is None and e0['after'] == 21.7),
])
C.group('E 非法/未知/无理由 → 报错不静默 + 快照不变（D5/I6）', [
    ('t_max=-5 被拒（InvalidValue）', rejected),
    ('被拒后快照逐位不变（自愈回退）', snap_unchanged),
    ('未知键被拒（UnknownKey）', unk_ok),
    ('空 reason 被拒（M1 归因必填）', reason_ok),
])
C.group('F undo 可逆：恢复 before + 发反向事件', [
    ('undo 后 t_max 从快照消失（恢复到未设）',
     'dynamic.t_max' not in s.effective()),
    ('undo 回执 restored/readback 一致', und['restored'] == und['readback']),
    ('undo 产生 kind=undo 的反向事件', und_ev['kind'] == 'undo'),
])

# ---------------- G. bus：指针 + 体积闸 + append-only ----------------
b = Bus(root=tmp)
arr = np.arange(2000).reshape(20, 100)
ev_ptr = b.append(actor='eng', kind='run', target='r1', after=arr, reason='落盘')
ptr = ev_ptr['after']
ev_big = b.append(actor='eng', kind='run', target='r2',
                  after={'blob': 'x' * 20000}, reason='超体积')
tr = ev_big['after']
b2 = Bus(root=tmp)                       # 模拟重启：从盘重载
C.group('G bus：I3 指针+sha1 / I4 体积闸告知 / M4 append-only 续 seq', [
    ('大数组不内联，转指针（$pointer + shape + sha1）',
     ptr.get('$pointer') is True and ptr.get('shape') == [20, 100]
     and len(ptr.get('sha1', '')) == 40),
    ('指针落 npz 到 payloads/', os.path.exists(
        os.path.join(tmp, ptr['path'])) if 'path' in ptr else False),
    ('超体积文本被截断且**告知截了多少+原文在哪**',
     tr.get('$truncated') is True and tr.get('total', 0) > tr.get('kept', 0)
     and 'where' in tr),
    ('append-only：重启后 last_seq 续上、事件全在',
     b2.last_seq == b.last_seq and len(b2) == len(b)),
    ('since(seq) 只回增量（diff-since-seq，R1）',
     len(b2.since(b2.last_seq - 1)) == 1),
])

# ---------------- H. I1 无侧门（AST 守卫 + 致死性证明） ----------------
def _snapshot_writes(tree):
    """返回 [(method_qualname, lineno)]：所有改 `_snapshot` 的语句位置。

    抓：`self._snapshot = ...`（重绑）、`self._snapshot[k] = ...`（下标写）、
    `del self._snapshot[k]`、`self._snapshot.pop/update/clear/setdefault(...)`。
    """
    hits = []

    def _is_snap_attr(node):
        return isinstance(node, ast.Attribute) and node.attr == '_snapshot'

    def _scan_body(body, qual):
        for nd in ast.walk(ast.Module(body=list(body), type_ignores=[])):
            # self._snapshot = X  /  self._snapshot[k] = X
            if isinstance(nd, ast.Assign):
                for tg in nd.targets:
                    if _is_snap_attr(tg):
                        hits.append((qual, nd.lineno))
                    if (isinstance(tg, ast.Subscript)
                            and _is_snap_attr(tg.value)):
                        hits.append((qual, nd.lineno))
            if isinstance(nd, ast.Delete):
                for tg in nd.targets:
                    if (isinstance(tg, ast.Subscript)
                            and _is_snap_attr(tg.value)):
                        hits.append((qual, nd.lineno))
            if isinstance(nd, ast.Call) and isinstance(nd.func, ast.Attribute):
                if (nd.func.attr in ('pop', 'update', 'clear', 'setdefault',
                                     'popitem', '__setitem__')
                        and _is_snap_attr(nd.func.value)):
                    hits.append((qual, nd.lineno))

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    _scan_body(sub.body, '%s.%s' % (node.name, sub.name))
        elif isinstance(node, ast.FunctionDef):
            _scan_body(node.body, node.name)
        else:
            _scan_body([node], '<module>')
    return hits


WHITELIST = {'Session.__init__', 'Session._write'}   # 唯一允许改快照的两处
violations, other_mod_refs = [], []
for fn in sorted(os.listdir(_MECHA)):
    if not fn.endswith('.py'):
        continue
    tree = ast.parse(open(os.path.join(_MECHA, fn), encoding='utf-8').read())
    for qual, lineno in _snapshot_writes(tree):
        if fn != 'session.py' or qual not in WHITELIST:
            violations.append('%s:%s:%d' % (fn, qual, lineno))
    # 除 session.py 外，任何模块都不该**引用** _snapshot（连读都不该绕过接口）
    if fn != 'session.py':
        for nd in ast.walk(tree):
            if isinstance(nd, ast.Attribute) and nd.attr == '_snapshot':
                other_mod_refs.append('%s:%d' % (fn, nd.lineno))

# 致死性证明：扫描器对一段"裸写入"必须亮红（否则它是永不触发的哑守卫，D4）
_ROGUE = ('class Session:\n'
          '    def rogue_backdoor(self, k, v):\n'
          '        self._snapshot[k] = v\n')           # 绕过 set 的侧门
_rogue_hits = _snapshot_writes(ast.parse(_ROGUE))
lethal_fires = any(q not in WHITELIST for q, _ in _rogue_hits)

C.group('H I1 无侧门：改 _snapshot 只在 __init__/_write（AST 守卫 + 致死性）', [
    ('mecha 内改 _snapshot 的语句都在白名单（违规：%s）' % (violations or '无'),
     not violations),
    ('session.py 之外无模块引用 _snapshot（残留：%s）' % (other_mod_refs or '无'),
     not other_mod_refs),
    ('**致死性**：扫描器对故意裸写入 rogue_backdoor 会亮红',
     lethal_fires and len(_rogue_hits) >= 1),
])

sys.exit(0 if C.summary() else 1)
