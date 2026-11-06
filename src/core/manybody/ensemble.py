# -*- coding: utf-8 -*-
"""Monte-Carlo ensemble driver for many-body runs.

职责边界：
- 只做“抽样 → 跑既有 build_run/evolve_run → 汇总”；不实现第二种演化器。
- sampler 是 Python callable（``rng → calib``），因此这是研究 API，不是 MCP 工具。
- 位置/初态缺陷由 sampler 写进 calib；速度若只是 shot 间初始位置偏移，也由 sampler
  折进 pos。真正的时变位置/Doppler 属于新的物理模型，本模块不伪造。

结果区分 ``unconditional`` 与 ``conditional``；后选均值不得冒充无条件均值。
"""
import copy
import numpy as np

__all__ = ['run_ensemble', 'ensemble_stats']


def _path(value, path):
    """按键路径取值；先整键匹配（JSON 键可以含点），再逐层下钻。"""
    if isinstance(value, dict) and path in value:
        return value[path]
    node = value
    for part in str(path).split('.'):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _finite(value):
    try:
        return bool(np.isfinite(np.asarray(value, dtype=float)).all())
    except (TypeError, ValueError):
        return False


def ensemble_stats(values, *, ddof=1):
    """(n_samples, ...) 数组 → mean/stderr/n；n<2 的 stderr 用 NaN，不填 0。"""
    arr = np.asarray(values)
    n = int(arr.shape[0]) if arr.ndim else 0
    if n == 0:
        return {'mean': None, 'stderr': None, 'n': 0}
    mean = np.mean(arr, axis=0)
    if n < 2:
        stderr = np.full(np.shape(mean), np.nan, dtype=float)
    else:
        stderr = np.std(arr, axis=0, ddof=ddof) / np.sqrt(n)
    return {'mean': mean, 'stderr': stderr, 'n': n}


def run_ensemble(params, sampler, *, n_samples, seed=0, observables=('pops_atom',),
                 conditional=None, return_samples=False, cache=None):
    """对 ``sampler(rng)`` 产生的 calib 重复跑同一 RunSpec，并汇总观测。

    参数
    ----
    params : dict
        当前完整快照（至少含 ``manybody.calib``）；每个样本只替换 calib。
    sampler : callable
        ``sampler(rng) -> calib`` 或 ``(calib, overrides)``；返回的新 calib 会经
        ``validate_calib``。``overrides`` 可逐样本改 ``manybody.psi0`` 等运行键
        （例如初态翻转）。
    n_samples : int
        样本数（>=1）。
    seed : int
        随机种子；所有样本共享一条 ``default_rng(seed)`` 流，可复现。
    observables : sequence[str]
        结果键路径，如 ``('pops_atom', 'corr', 'observables.pauli_strings')``。
    conditional : callable | None
        ``conditional(result) -> bool``；返回后选统计，同时保留无条件统计。
    return_samples : bool
        是否附带每个有效样本的观测值（默认不带，避免大数组进上下文）。
    cache : object | None
        透传给 ``build_run/evolve_run``；显式 calib 通常可为 None。

    返回
    ----
    dict:
      ``n_samples / n_valid / seed / observables``
      ``unconditional = {mean, stderr, n}``
      ``conditional = {mean, stderr, n, acceptance}`` 或 None
      ``rejected``、``error_report``、可选 ``samples``

    注意：``stderr`` 是样本标准差（ddof=1）/√n；n=1 时为 NaN。
    """
    from . import runspec_mb as mb
    from .. import protocol

    if not callable(sampler):
        raise TypeError('run_ensemble: sampler 必须是 callable')
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError('run_ensemble: n_samples 必须 >= 1')
    if not isinstance(params, dict):
        raise TypeError('run_ensemble: params 必须是 dict')
    keys = tuple(observables)
    if not keys:
        raise ValueError('run_ensemble: observables 不能为空')
    n_atoms = len((params.get('manybody.calib') or {}).get('atoms') or [])
    if n_atoms == 0:
        raise ValueError('run_ensemble: params 缺少 manybody.calib.atoms')
    if conditional is not None and not callable(conditional):
        raise TypeError('run_ensemble: conditional 必须是 callable 或 None')

    rng = np.random.default_rng(int(seed))
    good, cond_values, rows, errors = [], [], [], []
    for index in range(n_samples):
        try:
            sample = sampler(rng)
            overrides = {}
            if isinstance(sample, tuple) and len(sample) == 2:
                sample, overrides = sample
            sample = mb.validate_calib(sample)
            if len(sample['atoms']) != n_atoms:
                raise ValueError('sampler 改变了原子数：%d ≠ %d'
                                 % (len(sample['atoms']), n_atoms))
            if not isinstance(overrides, dict):
                raise TypeError('sampler 的第二项必须是参数覆盖 dict')
            p = copy.deepcopy(params)
            p['manybody.calib'] = sample
            p.update(overrides)
            run = protocol.build_run(cache, p)
            res = protocol.evolve_run(run, cache=cache)
            values = {key: _path(res, key) for key in keys}
            if any(v is None for v in values.values()):
                missing = [k for k, v in values.items() if v is None]
                raise ValueError('结果缺少观测：%s' % ', '.join(missing))
            if any(not _finite(v) for v in values.values()):
                raise ValueError('观测含非有限值')
            good.append(values)
            if conditional is not None:
                cond_values.append(bool(conditional(res)))
            if return_samples:
                rows.append({'index': index,
                             'observables': {k: np.asarray(v).tolist()
                                             for k, v in values.items()}})
        except Exception as exc:                            # noqa: BLE001
            errors.append({'index': index, 'kind': type(exc).__name__,
                           'message': str(exc)[:300]})

    n_valid = len(good)
    out = {
        'n_samples': n_samples, 'n_valid': n_valid, 'seed': int(seed),
        'observables': list(keys), 'rejected': n_samples - n_valid,
        'unconditional': {}, 'conditional': None,
        'error_report': {'rejected': errors[:50], 'n_rejected': len(errors)},
    }
    if n_valid:
        out['unconditional'] = {
            key: ensemble_stats([row[key] for row in good]) for key in keys
        }
    else:
        out['unconditional'] = {key: {'mean': None, 'stderr': None, 'n': 0}
                                for key in keys}
    if conditional is not None:
        selected = [row for row, keep in zip(good, cond_values) if keep]
        n_sel = len(selected)
        out['conditional'] = {
            'n': n_sel,
            'acceptance': (float(n_sel) / n_valid) if n_valid else float('nan'),
            'mean': {}, 'stderr': {},
        }
        if n_sel:
            stats = {key: ensemble_stats([row[key] for row in selected])
                     for key in keys}
            out['conditional'].update(
                mean={key: stats[key]['mean'] for key in keys},
                stderr={key: stats[key]['stderr'] for key in keys})
        else:
            out['conditional'].update(
                mean={key: None for key in keys},
                stderr={key: None for key in keys})
    if return_samples:
        out['samples'] = rows
    return out
