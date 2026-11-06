# -*- coding: utf-8 -*-
"""近似后端的 Pauli 语义（R4）：经典自旋 = 因子化；轨迹 = 系综平均。

不把近似值冒充纯态期望：结果必须带 ``observables.approximation``。
PEPS 当前没有多站点 Pauli 收缩，能力表直接拒绝并给替代。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from checks.checklib import Checker
from src.core.manybody.backends import validate_time_capabilities
from src.core.manybody.classical_spin import evolve_classical_spin
from src.core.manybody.engine_trajectories import evolve_trajectories
from src.core.manybody.schema_mb import validate_pauli_strings

C = Checker('manybody_pauli_approx')


def main():
    n = 2
    cal = {'atoms': [{'id': i, 'pos': [float(i), 0.0, 0.0]} for i in range(n)],
           'pairs': [{'i': 0, 'j': 1, 'V': 0.0, 'J': 0.3, 'A': 0.0}]}
    specs = validate_pauli_strings([[[0, 'z']], [[0, 'x'], [1, 'x']]], n)
    t = np.linspace(0.0, 0.5, 3)

    classical = evolve_classical_spin(
        cal, psi0='rg', t_grid=t, Om=np.zeros(n, complex), delta=np.zeros(n),
        terms=['I2'], max_step_ns=0.05, pauli_strings=specs)
    obs = classical.get('observables') or {}
    spins = np.asarray(classical['spins'])
    C.group('经典自旋：因子化语义并显式标注', [
        ('有 pauli_strings', 'pauli_strings' in obs),
        ('形状 (n_t,n_strings)',
         np.asarray(obs.get('pauli_strings')).shape == (len(t), len(specs))),
        ('σᶻ₀ == 自旋 z 分量',
         np.allclose(np.asarray(obs['pauli_strings'])[:, 0].real,
                     spins[:, 0, 2], atol=1e-12)),
        ('approximation 标注 factorized',
         'factorized' in str(obs.get('approximation'))),
    ])

    traj = evolve_trajectories(
        cal, psi0='rg', t_grid=t, Om=np.zeros(n, complex), delta=np.zeros(n),
        terms=['I2'], n_trajectories=2, chi_max=16, truncerr=1e-10,
        max_step_ns=0.1, want_observables=True, seed=3, pauli_strings=specs)
    tobs = traj.get('observables') or {}
    C.group('轨迹：系综平均并显式标注', [
        ('有 pauli_strings', 'pauli_strings' in tobs),
        ('形状 (n_t,n_strings)',
         np.asarray(tobs.get('pauli_strings')).shape == (len(t), len(specs))),
        ('approximation 标注 ensemble',
         'ensemble' in str(tobs.get('approximation'))),
        ('stderr 仍存在（统计误差不被吞）', 'stderr' in traj),
    ])

    rejects = False
    try:
        validate_time_capabilities(
            'peps-real-time',
            pauli_strings=[{'label': 'x', 'terms': [(0, 'x')]}])
    except ValueError as exc:
        rejects = ('ed' in str(exc)) or ('tdvp' in str(exc))
    C.check('PEPS 拒绝时给出可用替代', rejects)
    return 0 if C.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
