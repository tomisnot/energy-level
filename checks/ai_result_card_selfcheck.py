# -*- coding: utf-8 -*-
"""自检：结果摘要 + 模型卡（D）——含"卡片必须在 summary 建好后挂载"的回归。

回归背景（2026-09-12）：模型卡块曾被插在 `summary = {...}` 之前却直接写
`summary['model_card']` → `UnboundLocalError`，**任何走 job_result 的运行都崩**
（首次真跑作业才暴露）。本文件就是它的闸：直接对 `_res_summary` 喂一个最小
结果 dict，断言返回的是 dict 且带 model_card。

**2026-09-14（#13 覆盖缺口修复）**：第 1 段（单体 `pops` 路径）过去在 headless
聚合里**永远 SKIP**（`_res_labels` 里 `_win().dynamic` 遇 None 抛 AttributeError），
于是"单体摘要"零覆盖、全绿是假象。现在照 `sublevel_trim_selfcheck.py` 的做法
在本脚本内建一个 **offscreen MainWindow 并 show()**（`_win()` 只认可见的
MainWindow），把该段变成**真断言**：喂 11×N 布居 → 断言终态/均值键**就是 GUI
真实态标签**（而不是 '#0' 这种退化键名）。

用法：python checks/ai_result_card_selfcheck.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')

from checks.checklib import Checker                          # noqa: E402

C = Checker('ai_result_card')
# ------------------- 计数模型：不变式包（见 checks/checklib.py）-------------------
# group(...) 开一个包，其后相邻的 check(...) 全部变成该包的**子条件**：整包只计 1
# 项，子条件逐个进 SUB=（覆盖面不缩水：SUB == 改前逐条计数的项数）。原断言名与判据
# 表达式一律原样保留——只是不再各自计 1 项；不过时整包自动展开子条件。
_PKG = None


class _Pkg:
    """一个不变式包（先缓冲子条件，关闭时一次性交给 C.group 登记）。"""

    def __init__(self, name, detail=''):
        self.name, self.detail, self.subs = name, detail, []

    def flush(self):
        C.group(self.name, self.subs, self.detail)


def group(name, detail=''):
    """开新的不变式包：此后 check() 的断言都记入它（直到下一个 group）。"""
    global _PKG
    if _PKG is not None:
        _PKG.flush()
    _PKG = _Pkg(name, detail)


def flush():
    """关闭最后一个包（本文件收尾）。"""
    global _PKG
    if _PKG is not None:
        _PKG.flush()
        _PKG = None


def check(name, cond, detail=''):
    """薄包装：判据表达式原样，只登记为**当前包的子条件**（detail 作现场值）。"""
    if _PKG is None:
        raise RuntimeError(f'check() 出现在任何 group() 之前：{name}')
    _PKG.subs.append((name + (('  | ' + str(detail)) if detail else ''),
                      bool(cond)))
    return bool(cond)


def chk(cond, msg):
    """薄包装（沿用本文件原签名）：登记为**当前包的子条件**，不再各自计 1 项。"""
    return check(str(msg), bool(cond))


try:
    from src.ai import server as S
except Exception as e:                              # noqa: BLE001
    print('  SKIP 无法 headless 导入 src.ai.server：%s' % e)
    C.skip('导入 src.ai.server', str(e))
    sys.exit(0 if C.summary() else 1)

import numpy as np                                   # noqa: E402

print('1) 单体结果摘要（pops 路径；本脚本自建 offscreen 窗口 → **真断言**）')
win = None
try:
    from PyQt5.QtWidgets import QApplication
    import app_entry as M
    _app = QApplication.instance() or QApplication(sys.argv)
    win = M.MainWindow()
    win.show()                       # `server._win()` 只认 isVisible() 的 MainWindow
    _app.processEvents()
except Exception as e:                               # noqa: BLE001
    print('  SKIP 构窗不可用（无 GUI 环境）：%s' % e)

if win is not None:
    from src.ai.sim_state import SimState
    st = SimState(win)
    grp = st.set_groups([(60, 0, 0.5), (6, 1, 1.5)])
    labels = [s.get('label', '?')
              for s in (getattr(win.dynamic, '_run_states', None) or [])]
    if not labels:                   # _run_states 只在 _run() 里填 → 手动按同一口径填
        win.dynamic._run_states = win.dynamic._current_states()
        labels = [s.get('label', '?') for s in win.dynamic._run_states]
    n = len(labels)
    group('A 单体结果摘要（pops 路径：model_card / 终态键 = GUI 真标签）')
    chk(S._win() is win, '_win() 认出了本脚本的可见 MainWindow（真跑，不是 SKIP）')
    chk(n >= 4, 'GUI 侧态表非空（建组 %s → %d 个态）' % (grp.get('ok'), n))
    res1 = {'t': np.linspace(0, 10, 11),
            'pops': np.abs(np.random.RandomState(0).rand(11, n)),
            'states_2d': None}
    out1 = S._res_summary(res1, save_npz=False, tag='t1')
    chk(isinstance(out1, dict), '返回 dict（未抛 UnboundLocalError）')
    chk('model_card' in out1, '单体摘要带 model_card')
    chk('final_pop' in out1, '单体摘要含 final_pop')
    chk('top_states_avg_pop' in out1, '单体摘要含 top_states_avg_pop')
    keys = list(out1.get('final_pop') or {})
    chk(bool(keys) and all(k in labels for k in keys),
        'final_pop 的键 = GUI 真实态标签（不是退化的 "#0"）：%s' % keys[:2])
    chk(len(out1.get('states') or []) >= min(n, 8),
        'states 清单取自 GUI（%d 条）' % len(out1.get('states') or []))
else:
    C.skip('A 单体结果摘要（pops 路径）', '构窗不可用（无 GUI 环境）')

print('2) 两体结果摘要（pops_2d 路径，标签取自 states_2d）')
group('B 两体结果摘要（pops_2d 路径，成对标签）')
s2 = [{'label': '(60P_{3/2} m_J=-0.5)×(60S_{1/2} m_J=+0.5)'},
      {'label': '(60S_{1/2} m_J=+0.5)×(60P_{3/2} m_J=-0.5)'}]
res2 = {'t': np.linspace(0, 10, 11),
        'pops_2d': np.abs(np.random.RandomState(1).rand(11, 2)),
        'states_2d': np.asarray(s2, dtype=object)}
out2 = S._res_summary(res2, save_npz=False, tag='t2')
chk(isinstance(out2, dict) and 'model_card' in out2, '两体摘要带 model_card')
chk(out2.get('two_body') is True, 'two_body 标记正确')
labs = list(out2.get('final_pop', {}).keys())
chk(any('60P_{3/2}' in x for x in labs), '两体 final_pop 用成对标签：%s' % labs[:2])

print('3) 模型卡字段与警告判定')
mc = out2['model_card']
group('C 模型卡字段与警告判定')
chk(isinstance(mc, dict), 'model_card 是 dict')
chk('warnings' in mc and isinstance(mc['warnings'], list), '含 warnings 列表')
res3 = {'t': np.linspace(0, 1, 3), 'pops_2d': np.zeros((3, 2)),
        'states_2d': np.asarray(s2, dtype=object),
        'diag': {'meta': {'dim': 2, 'n_pairs': 0, 'tol': 1e-8},
                 'pairs': {'n_microwave': 0, 'max_abs_v_mhz': 0.0}}}
out3 = S._res_summary(res3, save_npz=False, tag='t3')
w = (out3.get('model_card') or {}).get('warnings') or []
chk(any('中间态' in x for x in w),
    '微波候选对为 0 → 点名"疑缺中间态"：%s' % (w[:1],))

flush()
sys.exit(0 if C.summary() else 1)
