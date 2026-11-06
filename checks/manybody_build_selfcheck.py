# -*- coding: utf-8 -*-
"""`manybody.build` = 构造器**真正用掉的入参**（不是备注）；回执由 calib 现算。

三条判据：
1. 有 / 无 / 改 build，运行结果逐位相同（它不进运行）；
2. 快照里的 build 与 `build_calib_from_file` 用的是同一份，指纹一致；
3. 回执（`build_receipt`）完全由 calib 推导——没有独立副本可漂。
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import protocol                                      # noqa: E402
from src.core.manybody import runspec_mb                           # noqa: E402
from mecha_v2.run_monitor import RunMonitorStore                   # noqa: E402

FAILED = []


def check(name, cond, detail=''):
    print(('  [PASS] ' if cond else '  [FAIL] ') + name + (('  ' + str(detail)) if detail and not cond else ''))
    if not cond:
        FAILED.append(name)


def calib(n=4):
    return {"atoms": [{"id": i, "pos": [float(i), 0.0, 0.0],
                       "delta_mhz": -0.1 * (i % 2)} for i in range(n)],
            "pairs": [{"i": i, "j": i + 1, "J": 1.0} for i in range(n - 1)]}


def build(**over):
    b = {"source": "file", "file": "m.json"}
    b.update(over)
    return b


def snap(with_build=True, mutate=False):
    s = {"dynamic.mode": 2, "manybody.calib": calib(), "manybody.want": "gs",
         "manybody.eq_backend": "ed", "manybody.charge_sector": 2}
    if with_build:
        s["manybody.build"] = build(file="other.json") if mutate else build()
        s["manybody.label"] = "改过的标题" if mutate else "N4 冒烟"
        s["manybody.notes"] = "改过的备注" if mutate else "只用于读卡"
    return s


def run(s):
    return protocol.evolve_run(protocol.build_run(protocol.make_cache(), s))


def main():
    a = run(snap(True))
    b = run(snap(False))
    c = run(snap(True, mutate=True))
    check('build 不进入运行：有/无 build 的 E0 相同',
          abs(float(a['E0']) - float(b['E0'])) < 1e-12, (a['E0'], b['E0']))
    check('label/notes/build 一起改，E0 仍不变',
          abs(float(a['E0']) - float(c['E0'])) < 1e-12, (a['E0'], c['E0']))
    check('build 不进入运行：占据逐位相同',
          np.array_equal(np.asarray(a['n_atom'], float), np.asarray(b['n_atom'], float))
          and np.array_equal(np.asarray(a['n_atom'], float), np.asarray(c['n_atom'], float)))

    spec = protocol.build_run(protocol.make_cache(), snap())
    check('RunSpec 带着 build（构造器的入参）',
          spec.get('build') == build(), spec.get('build'))
    check('RunSpec 带着 label/notes',
          spec.get('label') == 'N4 冒烟' and spec.get('notes') == '只用于读卡',
          (spec.get('label'), spec.get('notes')))
    receipt = spec.get('build_receipt') or {}
    summary = spec.get('calib_summary') or {}
    check('回执由 calib 现算：指纹 = 模型摘要指纹',
          receipt.get('fingerprint') == summary.get('fingerprint'),
          (receipt.get('fingerprint'), summary.get('fingerprint')))
    check('回执计数与模型一致',
          receipt.get('n_atoms') == 4 and receipt.get('n_pairs') == 3, receipt)
    check('回执没有 built_at 这类不可推导字段', 'built_at' not in receipt)

    view = RunMonitorStore._view(spec, snap())
    check('卡片视图带 build 与回执',
          view.get('build') == build() and (view.get('build_receipt') or {}).get('n_pairs') == 3)
    check('卡片视图带 label/notes',
          view.get('label') == 'N4 冒烟' and view.get('notes') == '只用于读卡',
          (view.get('label'), view.get('notes')))
    check('多体卡片没有 field（磁场只属单/两体）', view.get('field') is None, view.get('field'))

    # 同一份 build 交给构造器，必须得到同一个模型
    tmp = Path(tempfile.mkdtemp())
    os.environ["EL_CALIB_DIR"] = str(tmp)
    (tmp / "m.json").write_text(json.dumps(calib()), encoding="utf-8")
    built = runspec_mb.build_calib_from_file(build()["file"])
    check('build 是构造器入参：同一份文件得到同一指纹',
          runspec_mb.calib_fingerprint(built) == summary.get('fingerprint'),
          (runspec_mb.calib_fingerprint(built), summary.get('fingerprint')))

    print('[manybody_build] %s  FAIL=%d' % ('PASS' if not FAILED else 'FAIL', len(FAILED)))
    return 0 if not FAILED else 1


if __name__ == '__main__':
    raise SystemExit(main())
