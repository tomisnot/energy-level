# -*- coding: utf-8 -*-
"""只读观测通道自检（t83）：get_diag 摘要/明细 + AI 运行默认采诊断。

为什么单列：诊断是 AI 理解"为什么"的主通道（为什么布居零变化、为什么这次慢），
它的失效方式是"看起来有数据、其实缺项/错位"，必须拿**真实跑出来的 diag**
逐项核对（而不是造一个假 dict）。

覆盖：
  1. 摘要：meta/acize/photon/pairs/solver 五段齐全，且数值与运行事实一致；
  2. pairs 明细：按 |v| 降序、上限生效、态标签正确；
  3. steps 明细：dt 统计与降采样；
  4. sectors / H 明细：形状与时间点；
  5. 未知 scope → ok=False（不静默给错东西）；
  6. 无诊断时可读的错误提示（引导"切上帝模式或 run_compute(diag=True)"）；
  7. freeze 场景实证：三光子梯子第三级在 pairs 里可见（诊断能解释路径）；
  8. `_ai_wants_diag` → 极简模式下 AI 运行也采诊断。

用法：python checks\\ai_diag_test.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402
from arc import Rubidium87                                       # noqa: E402
from src.core import static_atom as core                         # noqa: E402
from src.core.atomic_levels import make_levels                   # noqa: E402
from src.core.waveforms import Wave, GaussianEnv                 # noqa: E402
from src.core.field_build import build_fields_exact              # noqa: E402
from src.core.evolve_mixed import evolve_mixed                   # noqa: E402
from src.ai import server as S                                   # noqa: E402

C = Checker('ai_diag')

# ---- freeze 场景（含三光子梯子三级）----
cache = core.ArcCache(Rubidium87())
SPEC = [
    dict(n=60, l=0, j=0.5, kind='mJ', m=+0.5),
    dict(n=60, l=0, j=0.5, kind='mJ', m=-0.5),
    dict(n=60, l=1, j=1.5, kind='mJ', m=+0.5),
    dict(n=60, l=1, j=1.5, kind='mJ', m=-0.5),
    dict(n=59, l=2, j=2.5, kind='mJ', m=-0.5),
    dict(n=59, l=2, j=2.5, kind='mJ', m=-1.5),
    dict(n=58, l=3, j=3.5, kind='mJ', m=-2.5),
    dict(n=58, l=4, j=4.5, kind='mJ', m=-3.5),
]
states = make_levels(cache, SPEC)
psi0 = np.zeros(len(states), complex)
for i, s in enumerate(states):
    if s['n'] == 60 and s['l'] == 1 and float(s.get('m', 0)) == -0.5:
        psi0[i] = 1.0
w = Wave('微波', 7500.0, GaussianEnv(200, 40), pol='σ-', intensity_mwcm2=1.0)
fields = build_fields_exact(cache, states, [w])
t = np.linspace(0.0, 400.0, 41)
res = evolve_mixed(states, fields, psi0, t, diag=True)
diag = res['diag']

# 注入 job 表（get_diag 的取数路径就吃这张表）
S._jobs.clear()
S._jobs['jT'] = {'started': 0.0, 'done': True, 'error': None, 'res': res}

# ---------------- 1. 摘要 ----------------
r = S._t_get_diag('')
C.check('1 摘要可取且来源正确', r.get('ok') and r.get('source') == '最近运行 jT',
        str(r)[:90])
sm = r.get('summary') or {}
C.check('2 五段齐全（meta/acize/photon/pairs/solver）',
        all(k in sm for k in ('meta', 'acize', 'photon', 'pairs', 'solver')),
        str(list(sm)))
C.check('3 meta 与运行事实一致（N/P/dim/t_end）',
        sm['meta']['N'] == len(states) and sm['meta']['P'] == 1
        and sm['meta']['dim'] == len(states)
        and abs(sm['meta']['t_end'] - 400.0) < 1e-9,
        str(sm['meta']))
C.check('4 pairs 统计 == 打包出的对数',
        sm['pairs']['n'] == len(diag['pairs']['a'])
        and sm['pairs']['n_microwave'] + sm['pairs']['n_optical_vdd']
        == sm['pairs']['n'],
        str(sm['pairs']))
C.check('5 solver 段有步长统计与归一偏差',
        sm['solver']['n_steps'] > 0 and sm['solver']['dt_ns'] is not None
        and sm['solver']['norm_dev'] is not None, str(sm['solver']))
C.check('6 标签只给前 8 个 + 总数',
        len(sm['labels']) == 8 and sm['n_labels'] == len(states),
        f"{len(sm['labels'])}/{sm['n_labels']}")

# ---------------- 2. pairs 明细 ----------------
rp = S._t_get_diag('pairs', limit=5)
C.check('7 pairs 明细：上限生效、按 |v| 降序',
        rp.get('ok') and rp['n_returned'] <= 5
        and all(rp['by_abs_v_desc'][i]['abs_v_mhz']
                >= rp['by_abs_v_desc'][i + 1]['abs_v_mhz']
                for i in range(len(rp['by_abs_v_desc']) - 1)),
        f"n_returned={rp.get('n_returned')}")
C.check('8 pairs 明细带态标签（不是裸索引）',
        rp['by_abs_v_desc'] and rp['by_abs_v_desc'][0]['label_a'] != '')

# ---------------- 3/4. steps / sectors / H ----------------
rs = S._t_get_diag('steps', limit=10)
C.check('9 steps 明细：dt 统计 + 降采样时间点',
        rs.get('ok') and rs['dt_ns_stats'] and len(rs['t_ns']) <= 10
        and rs['nfev'] > 0, str({k: rs.get(k) for k in ('nfev', 'n_acc')}))
rc = S._t_get_diag('sectors', limit=8)
C.check('10 sectors 明细：P/dim/D_full 统计',
        rc.get('ok') and rc['P'] == 1 and rc['D_full_stats'] is not None,
        str({k: rc.get(k) for k in ('P', 'dim')}))
rh = S._t_get_diag('H')
C.check('11 H 明细（不落盘时只给形状 + 提示）',
        rh.get('ok') and rh['shapes'] and 'save_npz' in rh.get('hint', ''),
        str(rh.get('shapes'))[:80])

# ---------------- 5/6. 错误路径 ----------------
C.check('12 未知 scope → ok=False（不静默给错东西）',
        S._t_get_diag('nonsense').get('ok') is False)
S._jobs.clear()
r0 = S._t_get_diag('')
C.check('13 无诊断时给出可操作的提示（提到上帝模式/diag=true）',
        r0.get('ok') is False and '上帝模式' in r0.get('msg', '')
        and 'diag' in r0.get('msg', ''), r0.get('msg', '')[:80])

# ---------------- 7. freeze 实证：诊断能解释"路径" ----------------
S._jobs['jT'] = {'started': 0.0, 'done': True, 'error': None, 'res': res}
rq = S._t_get_diag('pairs', limit=100)
_probe = {(int(x['a']), int(x['b'])): x for x in rq['by_abs_v_desc']}
# 三级梯子（态序：0/1=60S，2/3=60P，4/5=59D，6=58F，7=58G）
_ladder = [(3, 5), (5, 6), (6, 7)]
C.check('14 三光子梯子三级都在诊断里（能解释路径是否被删）',
        all(k in _probe for k in _ladder),
        f'缺 {[k for k in _ladder if k not in _probe]} / 有 {sorted(_probe)}')
C.check('14b 第三级 58F→58G 的耦合与失谐可见（|v|>1 GHz、Δ≈7.08 GHz）',
        (6, 7) in _probe and _probe[(6, 7)]['abs_v_mhz'] > 1000
        and abs(abs(_probe[(6, 7)]['dD_mhz']) - 420.68) < 1.0,
        str(_probe.get((6, 7))))

# ---------------- 8. AI 运行默认采诊断 ----------------
C.check('15 server 侧有 _ai_wants_diag 开关（人机成本分离）',
        '_ai_wants_diag' in open(
            os.path.join('src', 'ai', 'server.py'), encoding='utf-8').read()
        and '_ai_wants_diag' in open(
            os.path.join('src', 'ui', 'ui_dynamic.py'),
            encoding='utf-8').read())

sys.exit(0 if C.summary() else 1)
