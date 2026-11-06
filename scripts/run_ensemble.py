# -*- coding: utf-8 -*-
"""命令行系综驱动：位置抖动 + 可选初态翻转（Python API 的薄包装）。

示例：
  python scripts/run_ensemble.py --params params.json --samples 20 --seed 1 \
      --sigma-pos-um 0.01 --flip-prob 0.1

params.json 是当前快照（至少含 manybody.calib）。速度不是 calib 字段；若只是
shot 间初始位置偏移，用 --sigma-pos-um 表达；真正的时变运动不在本脚本范围。
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.manybody import run_ensemble


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True)
    ap.add_argument('--samples', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sigma-pos-um', type=float, default=0.0)
    ap.add_argument('--flip-prob', type=float, default=0.0)
    ap.add_argument('--observable', action='append', default=None)
    a = ap.parse_args()

    params = json.load(open(a.params, encoding='utf-8'))
    calib = params.get('manybody.calib') or {}
    atoms = calib.get('atoms') or []
    if not atoms:
        raise SystemExit('params 缺少 manybody.calib.atoms')
    n = len(atoms)
    base_psi = params.get('manybody.psi0')
    if not isinstance(base_psi, str) or len(base_psi) != n:
        base_psi = 'g' * n

    def sampler(rng):
        cal = copy.deepcopy(calib)
        if a.sigma_pos_um > 0:
            for atom in cal['atoms']:
                atom['pos'] = [float(atom['pos'][k]
                                   + rng.normal(0.0, a.sigma_pos_um))
                               for k in range(3)]
        overrides = {}
        if a.flip_prob > 0 and rng.random() < a.flip_prob:
            bits = list(base_psi)
            i = int(rng.integers(0, n))
            bits[i] = 'r' if bits[i] == 'g' else 'g'
            overrides['manybody.psi0'] = ''.join(bits)
        return cal, overrides

    obs = tuple(a.observable or ('pops_atom',))
    res = run_ensemble(params, sampler, n_samples=a.samples, seed=a.seed,
                       observables=obs)
    print(json.dumps(_plain(res), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
