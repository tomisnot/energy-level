# -*- coding: utf-8 -*-
"""自检：AI 作业的**落盘产物**（事故 #11 的能力补齐 + 命名修正，2026-09-12）。

背景两件事：
  1. 长两体作业跑完时进程被 Qt fail-fast 杀掉 → 结果随进程消失（"成果全丢"）⇒
     AI 作业 done 一到就自动落盘：`runtime/ai_results/job_<jid>.json`（摘要）+
     `runtime/ai_results/ai_<jid>_<HHMMSS>.npz`（全量数组），且 `job_result` 即使
     `save_npz=False` 也回传该 npz 路径。
  2. 命名 bug（使用者实测）：曾写成 `ai_job{jid}_*`，而 jid 已含 "job" ⇒
     得到 `ai_jobjob3_210422.npz`。现统一为 `ai_{jid}_*`。

覆盖（不需要 Qt / 不需要 ARC / 不做演化——只喂一个假的 res 摘要）：
  A. `_res_summary(res, save_npz=True, tag='jobX')` 的 npz 名 = `ai_jobX_*.npz`
     （**没有** jobjob），且**真实落盘**（文件存在）→ 测完删掉
  B. `_dump_job_json(jid, res)` 返回 (json, npz)：json 名 `job_<jid>.json`、
     npz 名 `ai_<jid>_*.npz`；两者都存在；json 内容含 `npz` 与 `job_id`
  C. 命名唯一性：两次落盘不覆盖（HHMMSS 相同也由 jid 区分 / 且不带 jobjob）
  D. 源码级：`_dump_job_json` 里不再出现 `'job' + jid`

用法：python checks/ai_job_artifacts_selfcheck.py（毫秒级）
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402
from src.ai import server as S                                   # noqa: E402

C = Checker('ai_job_artifacts')
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
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_made = []


def fake_res():
    """假结果：**带 states_2d**（两体口径）→ `_res_labels` 不依赖窗口，可 headless 测。"""
    n_t = 5
    labs = ['(60S+0.5)x(60S+0.5)', '(60S+0.5)x(60P-0.5)',
            '(60P-0.5)x(60S+0.5)', '(60P-0.5)x(60P-0.5)']
    return {'t': np.linspace(0.0, 1.0, n_t),
            'pops_2d': np.tile(np.array([0.5, 0.25, 0.15, 0.10]), (n_t, 1)),
            'states_2d': [{'label': L, 'ket_i': i, 'ket_j': 0,
                           'E_mhz': 0.0} for i, L in enumerate(labs)],
            'method': 'mixed', 'backend': 'mixed-phi-dop853-pair'}


def _rm(p):
    try:
        if p and os.path.exists(p):
            os.remove(p)
    except OSError:
        pass


try:
    print('A) _res_summary 的 npz 命名（tag=jid）')
    sA = S._res_summary(fake_res(), save_npz=True, tag='jobZ')
    p = sA.get('npz')
    _made.append(p)
    group('A 单体 _res_summary：npz 命名（无 jobjob）+ 真实落盘')
    check('A1 npz 路径存在', bool(p) and os.path.exists(p), str(p))
    check('A2 文件名 = ai_jobZ_*.npz（**无 jobjob**）',
            p and os.path.basename(p).startswith('ai_jobZ_')
            and 'jobjob' not in os.path.basename(p),
            os.path.basename(p or ''))
    check('A3 落在 runtime/ai_results/ 下',
      p and os.path.basename(os.path.dirname(p)) == 'ai_results',
            str(p))

    print('B) _dump_job_json 返回 (json, npz) 且都落盘')
    jp, np_ = S._dump_job_json('jobZZ', fake_res())
    _made += [jp, np_]
    group('B _dump_job_json：返回 (json, npz) + 都落盘 + json 内容可读')
    check('B1 json 存在且名 = job_jobZZ.json',
            jp and os.path.exists(jp)
            and os.path.basename(jp) == 'job_jobZZ.json', str(jp))
    check('B2 npz 存在且名 = ai_jobZZ_*.npz（无 jobjob）',
            np_ and os.path.exists(np_)
            and os.path.basename(np_).startswith('ai_jobZZ_')
            and 'jobjob' not in os.path.basename(np_),
            os.path.basename(np_ or ''))
    d = json.load(open(jp, encoding='utf-8')) if jp else {}
    check('B3 json 内容含 job_id 与 npz 路径',
            d.get('job_id') == 'jobZZ' and d.get('npz') == np_, str(sorted(d))[:90])
    check('B4 json 摘要含 top_states_avg_pop（结果可读，不只是路径）',
            'top_states_avg_pop' in d)

    print('C) 命名不冲突')
    sC = S._res_summary(fake_res(), save_npz=True, tag='jobZZ')
    _made.append(sC.get('npz'))
    group('C 命名不冲突 + 源码级命名修正')
    check('C1 同名 jid 两次落盘仍是 ai_jobZZ_*（时间戳区分）',
            sC.get('npz') and os.path.basename(sC['npz']).startswith('ai_jobZZ_')
            and 'jobjob' not in os.path.basename(sC['npz']),
            os.path.basename(sC.get('npz') or ''))

    print('D) 源码级：不再拼 "job"+jid')
    src = open(os.path.join(ROOT, 'src', 'ai', 'server.py'),
               encoding='utf-8').read()
    check("D1 源码无 \"'job' + jid\"", "'job' + jid" not in src)
    check('D2 源码含 tag=jid', 'tag=jid' in src)
    flush()
finally:
    for _p in _made:
        _rm(_p)

ok = C.summary()
sys.exit(0 if ok else 1)
