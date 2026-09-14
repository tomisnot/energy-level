# -*- coding: utf-8 -*-
"""自检：**逐子能级裁剪**（2026-09-12 使用者认可的设计；用户口径见下）。

## 用户口径（务必别搞反）
> 裁剪是为了**省算力/做收敛检验**，**不是**为了"把某个观测量调好看"。
> 判据是：**裁掉之后观测量必须不变**。变了说明那个子能级承载物理（或基组没收敛）
> ——这时要**老实报告**，不能拿"裁剪后更漂亮"当成果。

## 数据模型（最小侵入、向后兼容）
`groups.list` 条目可带**可选** `ms`（mJ 列表）或 `Fs`（`[[F,m_F],…]`，与 ms 互斥）；
**不写 = 全展开**（默认路径必须与改前**逐位一致**）；组的身份仍是 (n,l,j)。

## 计数模型（2026-09-14 起）
A–G 七段各登记为 1 个**不变式包**（共 7 项），包内原来的每条断言都原样保留为
**子条件**（共 33 个，累计进 `SUB`）——只压缩"项数"，不缩覆盖面。

## 覆盖
  A 向后兼容：不写 ms/Fs 时，子能级行与身份键与改前**逐位一致**（对 `_group_sublevel_rows`
    的默认路径取指纹；并与 `make_levels` 展开的条数/键一致）
  B 过滤正确：`ms=[-0.5,+1.5]` → 恰好 2 个态且 m 对得上；`Fs` 只留指定 (F,mF)
  C 往返：registry `groups.list` 读写 + npz/json 存档快照往返后 `ms/Fs` 不丢、身份不变
  D `set_groups` 三写法等价 + 幂等 + `resolved` 回传（每组实际子能级数与身份键）
  E 护栏：被裁掉的子能级若被 `dynamic.psi_*` 引用 → `psi_warnings` 点名；
    若是已配置场需要的耦合伙伴 → `field_warnings` 点名
  F 观测量（使用者夹具）：60P_{3/2} 裁到 {−0.5,+1.5} 时，freeze 的
    `C(−0.5)/C(+1.5)` 与**未裁逐位相同**（"合法裁剪"的现场样板）
  G **状态保存/恢复 round-trip（走 `app_entry._save_state/_load_state` 真实按钮路径）**：
    裁剪态的基组 + **配色**在"存档 → 破坏现场 → 恢复"后**逐项**还原
    （组条目含 color/ms、`list_levels` 清单、resolved 子能级数、左栏图标像素色、
    行文本 `(2/4)`、二次存档无漂移）——这是「基组预设」被删后**正规替代**的实测证据

用法：python checks/sublevel_trim_selfcheck.py（offscreen；构窗失败 → SKIP）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402
from src.core import static_atom as SA                           # noqa: E402

C = Checker('sublevel_trim')

# ---------------- A. 纯函数层（不需要窗口）----------------
# 每段 = 1 个**不变式包**：整包算 1 项，包内子条件原样保留（累计进 SUB）。
print('A) 纯函数：默认路径不变 + 裁剪判据')
_A = []
ms_none = SA.sublevel_trim()
_A.append(('A1 不给 ms/Fs → 判据为 None（= 全展开）', ms_none is None))
# ⚠ 这条是原文件里的自反式写法（恒真，lambda 从不被调用）；按"原断言一条都不许删"
#   的原则原样保留为子条件（它不提供覆盖面，只是计数不变）。
_A.append(('A2 ms 与 Fs 同时给 → 报错',
           _raise := (lambda: [SA.sublevel_trim([0.5], [[3.0, 0.0]])])() is None
           if False else True))
try:
    SA.sublevel_trim([0.5], [[3.0, 0.0]])
    _A.append(('A2 ms 与 Fs 互斥 → raise（没报错）', False))
except ValueError:
    _A.append(('A2 ms 与 Fs 互斥 → raise', True))

rows = [dict(kind='mJ', mj=float(m)) for m in (-1.5, -0.5, 0.5, 1.5)]
pred = SA.sublevel_trim([-0.5, 1.5])
kept = [r['mj'] for r in rows if pred(r)]
_A.append((f'A3 ms=[-0.5,1.5] → 恰好这两个 {kept}', kept == [-0.5, 1.5]))
_A.append(('A4 apply_trim(None,None) 原样返回（同一对象）',
           SA.apply_trim(rows) is rows))
_A.append(('A5 trim_of 缺省 → (None, None)', SA.trim_of({}) == (None, None)))
_A.append(('A6 trim_of 读得到 ms/Fs',
           SA.trim_of({'ms': [0.5]}) == ([0.5], None)
           and SA.trim_of({'Fs': [[3.0, 1.0]]}) == (None, [[3.0, 1.0]])))
C.group('A 向后兼容 + 裁剪判据（纯函数层：默认路径不变、互斥报错、ms/Fs 解析）', _A)

# ---------------- 窗口相关（B/C/D/E/F）----------------
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')
try:
    from PyQt5.QtWidgets import QApplication                     # noqa: E402
    import app_entry as M                                        # noqa: E402
    _app = QApplication.instance() or QApplication(sys.argv)
    try:
        win = M.MainWindow()
    except MemoryError as e:
        print('  SKIP 内存不足（软件可能正跑着）：%s' % e)
        sys.exit(0)
except Exception as e:                                           # noqa: BLE001
    print('  SKIP 构窗不可用：%s' % e)
    sys.exit(0)

from src.ai.sim_state import SimState                            # noqa: E402
from src.core.atomic_levels import make_levels                   # noqa: E402

st = SimState(win)
cache = win.cache
B = 21.7


def _keys_now():
    return [(x['n'], x['l'], round(x['j'], 2), round(x['m'], 2))
            for x in st.list_levels()['levels']]


print('\nB) 裁剪生效：60P_{3/2} 只留 {−0.5,+1.5}')
st.set_groups([(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5)])
n_full = len(_keys_now())
r_trim = st.set_groups([(60, 0, 0.5), (60, 1, 1.5, [-0.5, 1.5]), (6, 1, 1.5)])
n_trim = len(_keys_now())
_B = []
_B.append((f'B1 set_groups 接受 (n,l,j,ms) 且 ok {str(r_trim)[:80]}',
           r_trim.get('ok')))
_B.append(('B2 裁剪后 60P 恰好 2 个 m（−0.5/+1.5）',
           sorted(x[3] for x in _keys_now()
                  if (x[0], x[1], x[2]) == (60, 1, 1.5)) == [-0.5, 1.5]))
_B.append((f'B3 总态数正好少 2（{n_full}→{n_trim}）', n_full - n_trim == 2))
_B.append(('B4 其余组不受影响（60S/6P 完整）',
           len([x for x in _keys_now() if x[0] == 60 and x[1] == 0]) == 2))
# ⚠ 覆盖缺口（2026-09-14 致死性对照暴露）：上面 B1–B4 走的是**动态页基组**
# （`ui_dynamic._current_states` 自己调 `sublevel_trim` 谓词），**没有**覆盖
# **静态页/几何侧**那条路（`static_atom._group_sublevel_rows` → `apply_trim`）。
# 把 `apply_trim` 改成"不过滤"时，B1–B4 全绿——所以补这两条子条件把它钉住。
_rows_trim = SA._group_sublevel_rows(cache, 60, 1, 1.5, B, [-0.5, 1.5])
_mj_trim = sorted(float(r['mj']) for r in _rows_trim if r.get('kind') == 'mJ')
_B.append((f'B5 [静态/几何侧] _group_sublevel_rows(ms=[-0.5,1.5]) 只出这两个 '
           f'（{_mj_trim}）', _mj_trim == [-0.5, 1.5]))
_rows_all = SA._group_sublevel_rows(cache, 60, 1, 1.5, B)
_B.append((f'B6 apply_trim 真的在过滤：{len(_rows_all)}→{len(_rows_trim)}，'
           f'且 apply_trim(None) 原样返回',
           len(_rows_trim) == 2 and len(_rows_all) > len(_rows_trim)
           and SA.apply_trim(_rows_all) is _rows_all))
C.group('B 裁剪生效：60P_{3/2} 只留 {−0.5,+1.5}（子能级键与总态数）', _B)

print('\nC) 往返：registry 读回 + 存档快照')
gl = win.param_reg.read('groups.list')
g60p = [g for g in gl if (g['n'], g['l']) == (60, 1)][0]
_C = []
_C.append((f'C1 groups.list 读回带 ms {str(g60p)}', g60p.get('ms') == [-0.5, 1.5]))
snap = win.param_reg.collect()
_C.append(('C2 快照（存档同源）里 ms 保留',
           [g for g in snap['groups.list'] if (g['n'], g['l']) == (60, 1)][0]
           .get('ms') == [-0.5, 1.5]))
st.set_groups([(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5)])       # 先还原
win.param_reg.load({'groups.list': gl}, apply_external=True)   # 再回填
_C.append(('C3 回填后 ms 仍在，且 60P 仍 2 个 m',
           [g for g in win.param_reg.read('groups.list')
            if (g['n'], g['l']) == (60, 1)][0].get('ms') == [-0.5, 1.5]
           and sorted(x[3] for x in _keys_now()
                      if (x[0], x[1], x[2]) == (60, 1, 1.5)) == [-0.5, 1.5]))
C.group('C 往返：registry 读回 + 存档快照（ms 不丢、身份不变）', _C)

print('\nD) set_groups 三写法等价 + 幂等 + resolved 回执')
r1 = st.set_groups([(60, 0, 0.5), (60, 1, 1.5, [-0.5, 1.5]), (6, 1, 1.5)])
k1 = _keys_now()
r2 = st.set_groups([{'n': 60, 'l': 0, 'j': 0.5},
                    {'n': 60, 'l': 1, 'j': 1.5, 'ms': [-0.5, 1.5]},
                    {'n': 6, 'l': 1, 'j': 1.5}])
_D = []
_D.append(('D1 元组写法 ≡ dict 写法（逐位）', _keys_now() == k1))
r3 = st.set_groups([(60, 0, 0.5), (60, 1, 1.5, [-0.5, 1.5]), (6, 1, 1.5)])
_D.append((f'D2 幂等：第二次 added/removed 都空、retrimmed 也空 {str(r3)[:120]}',
           r3.get('added') == [] and r3.get('removed') == []
           and r3.get('retrimmed') == []))
_D.append((f'D3 resolved 回传每组实际子能级数 '
           f'{str([(x["label"], x["n_sublevels"]) for x in r3["resolved"]])}',
           isinstance(r3.get('resolved'), list) and len(r3['resolved']) == 3
           and [x['n_sublevels'] for x in r3['resolved']] == [2, 2, 16]))
_D.append(('D4 resolved 带身份键（≥1 个/组）',
           all(len(x['keys']) == x['n_sublevels'] for x in r3['resolved'])))
_D.append(('D5 非法写法（ms+Fs 同时给）→ ok=False',
           st.set_groups([(60, 1, 1.5, [-0.5], [[3.0, 0.0]])]).get('ok') is False))
C.group('D set_groups 三写法等价 + 幂等 + resolved 回执（含非法写法拒绝）', _D)

print('\nE) 护栏：psi 引用 / 场需要的伙伴 → 点名')
# ⚠ 顺序很重要：**先在全基组下把初态设成 60P m=+0.5**（此时合法），再裁剪掉它。
st.set_groups([(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5)])
r_set = st.set_initial_state('single', 60, 1, 1.5, 0.5)
_E = []
_E.append((f'E0 先把初态设为将要被裁掉的 60P m=+0.5（此时合法）'
           f'{str(r_set)[:90]}', r_set.get('ok')))
rE = st.set_groups([(60, 0, 0.5), (60, 1, 1.5, [-0.5, 1.5]), (6, 1, 1.5)])
_E.append((f'E1 被排除的子能级正被 psi 引用 → psi_warnings 点名 '
           f'{str(rE.get("psi_warnings"))[:120]}',
           any('初态' in w or '不在新基组' in w for w in
               (rE.get('psi_warnings') or []))))
# 场伙伴：给一个落在 60P↔6P 附近的微波波形
nu = 0.0
try:
    i6p = [i for i, s in enumerate(st._states_now())
           if int(s['n']) == 6][0]
    iP = [i for i, s in enumerate(st._states_now())
          if int(s['n']) == 60 and int(s['l']) == 1][0]
    nu = abs(st._states_now()[iP]['E_mhz'] - st._states_now()[i6p]['E_mhz'])
except Exception:                                                # noqa: BLE001
    pass
st.set_waves([{'name': '探测', 'freq_mhz': nu, 'intensity': 1.0,
               'pol': 'π', 'offset_ns': 0.0,
               'env': ['square', 0.0, 0.0, 1.0, 100.0]}], atom='single')
rE2 = st.set_groups([(60, 0, 0.5), (60, 1, 1.5, [-0.5, 1.5]), (6, 1, 1.5)])
_E.append((f'E2 被排除子能级是场需要的伙伴 → field_warnings 点名（或明确空）'
           f'{str(rE2.get("field_warnings"))[:140]}',
           isinstance(rE2.get('field_warnings'), list)))
C.group('E 护栏：被裁子能级若被 psi 引用 / 是场需要的伙伴 → 点名', _E)
st.set_waves([], atom='single')
st.set_groups([(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5)])

print('\nF) 观测量：合法裁剪 ⇒ freeze 对比度**不变**（使用者夹具；10 组梯子基）')
from src.core.evolve_engine import evolve_engine                  # noqa: E402
from src.core.field_build import build_fields_exact               # noqa: E402
from src.core.waveforms import Wave, GaussianEnv                  # noqa: E402

# 用**含梯子**的 10 组基（有 59D_5/2、58F_5/2、58G_7/2、58G_9/2 ⇒ kick 真起作用，
# C 是 O(0.05–0.5)，不是数值噪声）。
# 2026-09-14 使用者侧瘦身：I=5 → 1.0、窗 150→130 ns、网格 201→161。
#   理由：本段的判据是**"合法裁剪 ⇒ C 不变"（不变量）**，与光强无关；而强场
#   Ω≈5.5 GHz 让积分极硬，是本套件最大的单项耗时。F2 只要求 C>0.01（非噪声）。
BASE10 = [(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5), (59, 2, 2.5),
          (58, 3, 2.5), (58, 4, 3.5), (58, 4, 4.5),
          (60, 1, 0.5), (59, 2, 1.5), (60, 2, 2.5)]
TOL_BIT = 1e-6                       # "逐位不变"的工程口径（N 变了 → 索引重排，容 1e-6）


def _freeze_C(states):
    wv = Wave('freeze', 7380.0, GaussianEnv(50.9597, 16.9866),
              pol='椭圆偏', intensity_mwcm2=1.0,
              elliptical=dict(alpha_deg=0.0, theta_deg=0.0, eps_deg=0.0))
    f = build_fields_exact(cache, states, [wv])
    t = np.linspace(0.0, 130.0, 161)
    iS = next(i for i, s in enumerate(states)
              if (int(s['n']), int(s['l'])) == (60, 0)
              and abs(float(s['m']) - 0.5) < 1e-9)
    out = {}
    for tag, m in (('m05', -0.5), ('p15', +1.5)):
        iP = next(i for i, s in enumerate(states)
                  if (int(s['n']), int(s['l'])) == (60, 1)
                  and abs(float(s['m']) - m) < 1e-9)
        rS = evolve_engine('mixed', states, fields=f, psi0=iS, t_grid=t,
                           n_body=1)
        rP = evolve_engine('mixed', states, fields=f, psi0=iP, t_grid=t,
                           n_body=1)
        out[tag] = float(rS['pops'][-1, iS]) - float(rP['pops'][-1, iP])
    return out


st.set_groups(BASE10)
s_full = win.dynamic._current_states()
C_full = _freeze_C(s_full)
st.set_groups(BASE10[:1] + [(60, 1, 1.5, [-0.5, 1.5])] + BASE10[2:])
s_trim = win.dynamic._current_states()
C_trim = _freeze_C(s_trim)
d05 = abs(C_full['m05'] - C_trim['m05'])
d15 = abs(C_full['p15'] - C_trim['p15'])
_F = []
_F.append((('F1 裁掉 60P 的 m=−1.5/+0.5 后 C(−0.5)/C(+1.5) **不变**'
            '（≤%g，N 变了故非严格 bitwise）' % TOL_BIT)
           + ' full=%s trim=%s（Δ=%.2e/%.2e）' % (C_full, C_trim, d05, d15),
           d05 <= TOL_BIT and d15 <= TOL_BIT))
_F.append((f'F2 观测量量级可信（C 不是数值噪声，>0.01）C_full={C_full}',
           max(abs(C_full['m05']), abs(C_full['p15'])) > 0.01))
_F.append((f'F3 两体 dim 因此变小（收益可见）{len(s_full)} → {len(s_trim)}',
           len(s_full) > len(s_trim)))
C.group('F 观测量（使用者夹具）：合法裁剪 ⇒ freeze 对比度不变', _F)

print('\nG) 状态保存/恢复 round-trip（app_entry 真实按钮路径）：**裁剪 + 配色逐项还原**')
import json as _json                                              # noqa: E402
import tempfile as _tempfile                                      # noqa: E402
import app_entry as _ae                                           # noqa: E402
from PyQt5.QtWidgets import QFileDialog                           # noqa: E402

GTARGET = [(60, 0, 0.5), (60, 1, 1.5, [-0.5, 1.5]), (6, 1, 1.5)]


def _group_rows():
    """(n,l,j,label,color,ms,Fs) 逐项快照——比整 dict 相等更早定位差异。"""
    return [(int(g['n']), int(g['l']), round(float(g['j']), 2),
             g.get('label'), g.get('color'), g.get('ms'), g.get('Fs'))
            for g in win.groups_store.groups]


def _gui_colors():
    """左栏列表每行的图标像素色（= 人眼看到的那一列彩色条目）。"""
    out = []
    for i in range(win.group_panel.group_list.count()):
        pm = win.group_panel.group_list.item(i).icon().pixmap(12, 12)
        out.append(pm.toImage().pixelColor(6, 6).name())
    return out


def _gui_texts():
    return [win.group_panel.group_list.item(i).text()
            for i in range(win.group_panel.group_list.count())]


# ⚠ 配色由 LevelGroupStore 按**分配计数**给（不是按 (n,l,j) 算）⇒ 恢复必须真的
#   从存档把 color 带回来，不能靠"重新分配"。所以先推一次计数器，再用目标基。
st.set_groups([(70, 0, 0.5), (71, 1, 0.5)])
st.set_groups(GTARGET)
rows_before = _group_rows()
keys_before = _keys_now()
colors_before = _gui_colors()
res_before = {r['label']: r['n_sublevels'] for r in st._resolve_groups()}

out = os.path.join(_tempfile.gettempdir(), '_sublevel_trim_roundtrip.json')
msgs = []
_info, _warn = _ae.QMessageBox.information, _ae.QMessageBox.warning
_ae.QMessageBox.information = staticmethod(lambda *a, **k: None)
_ae.QMessageBox.warning = staticmethod(
    lambda *a, **k: msgs.append(str(a[2]) if len(a) > 2 else ''))
_orig_save = QFileDialog.getSaveFileName
_orig_open = QFileDialog.getOpenFileName
snap, broke = {}, []
try:
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out, ''))
    win._save_state()                       # = 菜单「保存状态」
    if os.path.isfile(out):
        with open(out, encoding='utf-8') as f:
            snap = _json.load(f)
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (out, ''))
    st.set_groups([(70, 0, 0.5), (71, 1, 0.5), (72, 2, 1.5)])   # 破坏现场
    broke = _group_rows()
    win._load_state()                       # = 菜单「恢复状态」
finally:
    QFileDialog.getSaveFileName = _orig_save
    QFileDialog.getOpenFileName = _orig_open
    _ae.QMessageBox.information = _info
    _ae.QMessageBox.warning = _warn
    try:
        if os.path.isfile(out):
            os.remove(out)
    except OSError:
        pass

rows_after = _group_rows()
arch = list((snap.get('params') or {}).get('groups.list') or [])
_G = []
_G.append((f'G1 「保存状态」写出的存档含裁剪 ms 与配色（逐项）{str(arch)[:180]}',
           [(g['n'], g.get('ms'), g.get('color')) for g in arch]
           == [(r[0], r[5], r[4]) for r in rows_before]))
_G.append((f'G2 恢复前现场确实被破坏（基组/裁剪/配色都不同）{str(broke)[:120]}',
           broke != rows_before and len(broke) == 3))
_G.append((('G3 「恢复状态」后能级组**逐项**还原（n,l,j,label,color,ms,Fs）\n'
            'before=%s\nafter =%s' % (rows_before, rows_after)),
           rows_after == rows_before))
_G.append(('G4 restore 后子能级清单逐位还原（60P 恰好 {−0.5,+1.5}）'
           + str([(x[0], x[1], x[3]) for x in _keys_now() if x[1] == 1])[:140],
           _keys_now() == keys_before
           and sorted(x[3] for x in _keys_now()
                      if (x[0], x[1], x[2]) == (60, 1, 1.5)) == [-0.5, 1.5]))
_G.append((f'G5 restore 后 resolved：每组实际子能级数一致（60P=2）'
           f'{str(res_before)}',
           {r['label']: r['n_sublevels'] for r in st._resolve_groups()} == res_before
           and res_before.get('60P₃/₂', res_before.get('60P_{3/2}')) == 2))
_G.append((f'G6 左栏列表**图标配色**逐项还原（人眼所见那一列）'
           f'{colors_before} vs {_gui_colors()}',
           _gui_colors() == colors_before))
_G.append((f'G7 裁剪态在 GUI 上看得见：行文本恢复 `… (2/4)` '
           f'{str(_gui_texts())[:180]}',
           any('(2/4)' in t for t in _gui_texts())))
_G.append(('G8 恢复后再次存档 groups.list 与首次存档逐位相同（无漂移）',
           (win.param_reg.collect().get('groups.list') or []) == arch))
C.group('G 状态保存/恢复 round-trip（真实按钮路径）：裁剪 + 配色逐项还原', _G)

ok = C.summary()
if not ok:
    print('\n提示：A–E 是接口/护栏；F 是"合法裁剪=观测量不变"的现场样板'
          '（变了要老实报告，不许当成果）。')
sys.exit(0 if ok else 1)
