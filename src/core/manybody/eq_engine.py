# -*- coding: utf-8 -*-
"""多体**平衡态**产品引擎：把 ED/GPU批量/Lanczos/KPM/DMRG 经唯一分派点接成"可调用"。

- `run_equilibrium(spec)`：`runspec.evolve_run` 在 `nbody=='many'` 且 `spec['equilibrium']` 时
  调它（分派表 method ∈ gs/thermal/dos）。吃紧凑 calib + Om/delta（**不含时**），经
  `eq_route.solve_equilibrium` 路由到后端，返回统一 Result（含 method/backend/states 契约键）。
- `build_equilibrium(cache, params, want=..., method=...)`：登记册快照 → 平衡态 RunSpec
  （复用 runspec_mb 的 model/calibrate/drive，**calibrate 碰 ARC ⇒ 主线程**）。
- `solve_equilibrium(cache, params, ...)`：build+run 一步，**产品/AI 可直接调的入口**。

任意点云几何都能被 ed/lanczos/kpm 直接吃（都从 build_H_sparse(calib) 出发）；
'dmrg' 后端仅当 spec 带 'ladder' 时可用。
"""
import numpy as np

from . import eq_route, runspec_mb as _mb

__all__ = ['build_equilibrium', 'run_equilibrium', 'solve_equilibrium']


def _dmrg_ok(calib, Om=None, delta=None, max_cross=12, min_legs=2):
    """该点云适合自动路由到 DMRG 吗？——看**跨过切割的最大键数**。

    自写 autoMPO 后 **任意几何都能 DMRG**（正确性与排序无关），本函数只为 `auto`
    把“DMRG 不划算”的情形挡掉：跨切割键数 → MPO 键维 → DMRG 代价。
    ❌ 不用“最大排程”做判据：蛇形排序下 2-leg 梯子的首个梯级键天然跳整条链
    （实测 2×4 排程=7），会误拒好的几何；而跳接的长程键反而排程短，会误接受。
    """
    try:
        from . import dmrg as _dm
        _dm._require_tenpy()
        order, _onsite, coups, info = _dm.terms_from_calib(calib, Om, delta)
        n = len(order)
        if n < 4 or info.get('n_legs', 0) < min_legs:
            return False
        if not coups:
            return True
        # 按**去重后的原子对**计：一个键会因 V / J+h.c. / A+h.c. 展开成最多 5 项，
        # 数项会高估代价近 5 倍（实测把好好的 2×4 梯子误拒）。
        bonds = {(int(a), int(b)) for (_co, a, _o1, b, _o2) in coups}
        cross = 0
        for k in range(n - 1):
            c = sum(1 for (a, b) in bonds if a <= k < b)
            cross = max(cross, c)
        return cross <= max_cross
    except Exception:
        return False


def build_equilibrium(cache, params, want='gs', method=None, betas=None,
                      num=400, nvec=1500, seed=3):
    """快照 → 平衡态 RunSpec（紧凑 calib + 驱动，无时间轴）。calibrate 碰 ARC → 主线程。"""
    p = params or {}
    warnings = []
    model = _mb.model_from_params(p, warnings=warnings)
    n = len(model.atoms)
    if n == 0:
        raise ValueError('多体平衡态：manybody.atoms 为空——先摆原子')
    second_order = bool(_mb.mb_get(p, 'manybody.second_order'))
    guard_mhz = float(_mb.mb_get(p, 'manybody.guard_mhz'))
    ed_max_dim = int(_mb.mb_get(p, 'manybody.ed_max_dim'))
    calib = _mb.calibrate_memo(cache, model, second_order=second_order,
                               guard_mhz=guard_mhz)
    Om, delta, gamma = _mb.drive_from_model(model)
    backend = (method or _mb.mb_get(p, 'manybody.eq_backend')
               or _mb.mb_get(p, 'manybody.solver') or 'auto')
    states = [{'label': 'atom%d' % a['id'], 'atom_id': a['id']} for a in calib['atoms']]
    dmrg_ok = _dmrg_ok(calib, Om, delta)
    return {
        'version': 1, 'nbody': 'many', 'equilibrium': True, 'two_body': False,
        'want': want, 'method': backend, 'method_in': backend, 'betas': betas,
        'calib': calib, 'n_atoms': n, 'dim': 1 << n,
        'Om': Om, 'delta': delta, 'gamma': gamma,
        'terms': list(model.terms), 'template': model.template,
        'kpm': dict(num=num, nvec=nvec, seed=seed),
        'dmrg': dict(chi_max=int(p.get('manybody.chi_max') or 200),
                     truncerr=float(p.get('manybody.truncerr') or 1e-10),
                     max_sweeps=int(p.get('manybody.max_sweeps') or 60)),
        'states': states, 'labels': [s['label'] for s in states],
        'warnings': warnings, 'ed_max_dim': ed_max_dim, 'dmrg_ok': dmrg_ok,
    }


def run_equilibrium(spec, log=None, progress_arr=None):
    """平衡态 RunSpec → 统一 Result（`evolve_run` 的 many+equilibrium 分支入口）。"""
    log = log or (lambda *a: None)
    want = spec.get('want', 'gs')
    n = int(spec.get('n_atoms') or len(spec['calib']['atoms']))
    if progress_arr is not None:
        try:
            progress_arr[0] = 0.3
        except Exception:
            pass
    system = dict(calib=spec['calib'], Om=spec.get('Om'), delta=spec.get('delta'),
                  n_sites=n, is_ladder=bool(spec.get('ladder')),
                  ladder=spec.get('ladder'), batch=spec.get('batch', 1),
                  dmrg_ok=bool(spec.get('dmrg_ok')))
    kp = spec.get('kpm') or {}
    dm = spec.get('dmrg') or {}      # DMRG 截断控制（不传则用下面的默认值）
    res = eq_route.solve_equilibrium(
        system, want=want, betas=spec.get('betas'), backend=spec.get('method', 'auto'),
        num=kp.get('num', 400), nvec=kp.get('nvec', 1500), seed=kp.get('seed', 3),
        chi_max=int(dm.get('chi_max', 200)),
        truncerr=float(dm.get('truncerr', 1e-10)),
        max_sweeps=int(dm.get('max_sweeps', 60)),
        exact=bool(spec.get('exact', want == 'gs')))
    res['method'] = want
    res['states'] = spec.get('states')
    res['labels'] = spec.get('labels')
    res['n_atoms'] = n
    res['error_report'] = dict(warnings=list(spec.get('warnings') or []))
    log('[多体平衡态] want=%s → 后端 %s，n=%d' % (want, res.get('backend'), n))
    if progress_arr is not None:
        try:
            progress_arr[0] = 1.0
        except Exception:
            pass
    return res


def solve_equilibrium(cache, params, want='gs', method=None, betas=None):
    """产品/AI 直接入口：快照 → 平衡态 RunSpec → Result。"""
    return run_equilibrium(build_equilibrium(cache, params, want=want, method=method,
                                             betas=betas))
