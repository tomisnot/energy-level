# -*- coding: utf-8 -*-
"""耗散引擎契约修正回归判据（Re0 重构 · 第 14 刀 Phase 0）。

锁三条已在沙盒 core 可达、改前会"静默改物理 / 契约分叉"的问题：
  C1 sambe 含 decays 转 evolve_reference 时 **E_shift 被丢**（现须透传，
     且与"手工把该态 E_mhz 平移"逐位等价；旧行为是传/不传逐位相同）。
  C2 转参考路径返回**缺 error_report**（现须补，键含 max_norm_dev，
     后端以 reference- 打头；幺正/耗散两路都要有）。
  C5 hard 耗散显式落点的报错文案**指向不存在的 lindblad 引擎档**
     （现须改指 sambe，且仍抛 NotImplementedError）。

计数模型同 checklib：**不变式包**（整包 1 项，子条件累计 SUB）。
运行：D:\python310\python.exe checks\dissip_contract_selfcheck.py
"""
import os
import sys
import copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core.atomic_levels import make_levels, square
from src.core.evolve_util import decay, _single_loss_rates
from src.core.sambe import evolve_sambe
from src.core.evolve_reference import evolve_reference
from checklib import Checker

ck = Checker('dissip_contract_selfcheck')

# ---- 干净的二能级（不依赖 ARC：E_mhz 显式给定）----
E0, E1 = 0.0, 1000.0                 # MHz
OM = 60.0                            # 峰值 Rabi（MHz）
GAM = 10.0                           # 激发态自发辐射率（MHz，落点=丢失）
DELTA = 500.0                        # 被测 E_shift（MHz）


class _Cache:
    """make_levels 会预求值 cache.E_mhz 作默认（即使显式给 E_mhz）；
    本判据全部显式给定 E_mhz ⇒ 桩返回值被丢弃，仅为避开 ARC 导入。"""
    def E_mhz(self, n, l, j):
        return 0.0


def _states():
    return make_levels(_Cache(), [
        dict(n=60, l=0, j=0.5, m=-0.5, E_mhz=E0),
        dict(n=60, l=1, j=0.5, m=-0.5, E_mhz=E1)])


def _field():
    A = np.zeros((2, 2))
    A[0, 1] = A[1, 0] = OM
    return {'kind': 'exact', 'A': A, 'omega_L_mhz': E1 - E0,   # 共振
            'waveform': square(0.0, 1e12), 'label': '驱动'}


T = np.linspace(0.0, 200.0, 201)
DECAYS = [decay(1, GAM)]             # 激发态 → loss（走参考路径）

# ---------------- 包 1：C1 E_shift 透传 ----------------
subs1 = []
st = _states()
fld = _field()
with_shift = evolve_sambe(st, [fld], 1, T, decays=DECAYS,
                          E_shift=np.array([0.0, DELTA]))
no_shift = evolve_sambe(st, [fld], 1, T, decays=DECAYS)
# 手工平移等价：把 states[1] 的 E_mhz += DELTA，E_shift=None
st_shift = copy.deepcopy(st)
st_shift[1]['E_mhz'] += DELTA
manual = evolve_sambe(st_shift, [fld], 1, T, decays=DECAYS)
d_eq = float(np.max(np.abs(with_shift['pops'] - manual['pops'])))
d_eff = float(np.max(np.abs(with_shift['pops'] - no_shift['pops'])))
subs1.append((f'C1 E_shift 与手工平移 E_mhz 逐位等价 max|Δpops|={d_eq:.1e}',
              d_eq < 1e-15))
subs1.append((f'C1 E_shift 确有物理效果（≠ 不传）max|Δpops|={d_eff:.1e}',
              d_eff > 1e-3))
ck.group('C1 sambe+decays 透传 E_shift（逐位等价 + 非静默）', subs1,
         detail=f'eq={d_eq:.1e} eff={d_eff:.1e}')

# ---------------- 包 2：C2 转参考路径补 error_report ----------------
subs2 = []
er = with_shift.get('error_report')
subs2.append(('C2 含耗散（转参考）返回带 error_report', isinstance(er, dict)))
subs2.append(('C2 error_report 含 max_norm_dev',
              isinstance(er, dict) and 'max_norm_dev' in er))
subs2.append(('C2 后端以 reference- 打头',
              str(with_shift.get('backend')).startswith('reference-')))
subs2.append(('C2 max_norm_dev 有限且小',
              isinstance(er, dict) and np.isfinite(er['max_norm_dev'])
              and er['max_norm_dev'] < 1e-6))
# 幺正参考路径也应带 error_report
ref_u = evolve_reference(_states(), [_field()], 1, T)
subs2.append(('C2 幺正参考路径也带 error_report',
              isinstance(ref_u.get('error_report'), dict)
              and 'max_norm_dev' in ref_u['error_report']))
ck.group('C2 参考路径 error_report 契约', subs2)

# ---------------- 包 3：C5 报错文案改指真实档位 ----------------
subs3 = []
raised, msg = False, ''
try:
    _single_loss_rates(_states(), [decay(1, GAM, 0)])   # 显式落点 to=0（在基组内）
except NotImplementedError as e:
    raised, msg = True, str(e)
subs3.append(('C5 显式落点仍抛 NotImplementedError', raised))
subs3.append(('C5 文案不再含 "lindblad 引擎"', raised and 'lindblad 引擎' not in msg))
subs3.append(('C5 文案改指 sambe', raised and 'sambe' in msg))
ck.group('C5 hard 显式落点报错文案指向真实档位', subs3, detail=f'msg={msg!r}')

ok = ck.summary()
sys.exit(0 if ok else 1)
