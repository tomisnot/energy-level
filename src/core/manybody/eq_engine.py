# -*- coding: utf-8 -*-
"""多体平衡态产品适配：登记册快照 → RunSpec → eq_route。

build_equilibrium 只在 ARC cache 所属线程标定物理参数，run_equilibrium
消费数值快照，提供公共进度/取消检查并合并配置警告。数值算法和任务路由
由 eq_route 维护，参数默认值从 schema_mb 读取。
"""
import numpy as np

from . import eq_route, runspec_mb as _mb
from .hamiltonian_plan import build_plan, plan_summary
from .backends import normalize_equilibrium_backend
from .runtime import check_cancel, report_progress

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
                      num=None, nvec=None, seed=None):
    """快照 → 平衡态 RunSpec（紧凑 calib + 驱动，无时间轴）。calibrate 碰 ARC → cache 所属线程。"""
    p = params or {}
    warnings = []
    num = _mb.mb_get(p, 'manybody.kpm_num') if num is None else num
    nvec = _mb.mb_get(p, 'manybody.kpm_nvec') if nvec is None else nvec
    seed = _mb.mb_get(p, 'manybody.seed') if seed is None else seed
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
    plan = build_plan(calib, Om=Om, delta=delta, gamma=gamma,
                      terms=model.terms, constraint=p.get('manybody.constraint'), max_states=ed_max_dim)
    backend = normalize_equilibrium_backend(method or _mb.mb_get(p, 'manybody.eq_backend'), want)
    if betas is None:
        betas = _mb.mb_get(p, 'manybody.betas')
    if want == 'thermal' and betas is None:
        betas = [1.]
    states = [{'label': 'atom%d' % a['id'], 'atom_id': a['id']} for a in calib['atoms']]
    dmrg_ok = _dmrg_ok(calib, Om, delta)
    return {
        'version': 1, 'nbody': 'many', 'equilibrium': True, 'two_body': False,
        'want': want, 'method': backend, 'method_in': backend, 'betas': betas,
        'calib': calib, 'hamiltonian_plan': plan, 'plan_summary': plan_summary(plan),
        'n_atoms': n, 'dim': plan['dim'], 'full_dim': 1 << n,
        'charge_sector': p.get('manybody.charge_sector'),
        'return_state': bool(p.get('manybody.return_state', False)),
        'memory_budget_mb': p.get('manybody.memory_budget_mb'),
        'dtype': p.get('manybody.dtype') or 'float64',
        'device': _mb.mb_get(p, 'manybody.device'),
        'krylov_dim': _mb.mb_get(p, 'manybody.krylov_dim'),
        'Om': Om, 'delta': delta, 'gamma': gamma,
        'terms': list(model.terms), 'template': model.template,
        'kpm': dict(num=num, nvec=nvec, seed=seed),
        'qmc': dict(sweeps=int(_mb.mb_get(p, 'manybody.qmc_sweeps')),
                    burn_in=int(_mb.mb_get(p, 'manybody.qmc_burn_in')),
                    measure_every=int(_mb.mb_get(p, 'manybody.qmc_measure_every')),
                    M=p.get('manybody.qmc_M'), seed=seed),
        'dmrg': dict(chi_max=int(_mb.mb_get(p, 'manybody.chi_max')),
                     truncerr=float(_mb.mb_get(p, 'manybody.truncerr')),
                     max_sweeps=int(_mb.mb_get(p, 'manybody.max_sweeps'))),
        'states': states, 'labels': [s['label'] for s in states],
        'warnings': warnings, 'ed_max_dim': ed_max_dim, 'dmrg_ok': dmrg_ok,
    }


def run_equilibrium(spec, log=None, progress_arr=None):
    """平衡态 RunSpec → 统一 Result（`evolve_run` 的 many+equilibrium 分支入口）。"""
    log = log or (lambda *a: None)
    want = spec.get('want', 'gs')
    n = int(spec.get('n_atoms') or len(spec['calib']['atoms']))
    report_progress(progress_arr, 0.3)
    system = dict(calib=spec['calib'], plan=spec.get('hamiltonian_plan'),
                  Om=spec.get('Om'), delta=spec.get('delta'), gamma=spec.get('gamma'),
                  terms=spec.get('terms'),
                  n_sites=n, is_ladder=bool(spec.get('ladder')),
                  ladder=spec.get('ladder'), batch=spec.get('batch', 1),
                  dmrg_ok=bool(spec.get('dmrg_ok')),
                  charge_sector=spec.get('charge_sector'))
    system['memory_budget_mb'] = spec.get('memory_budget_mb')
    system['dtype'] = spec.get('dtype', 'float64')
    system['device'] = spec.get('device', 'cuda')
    system['krylov_dim'] = spec.get('krylov_dim', 48)
    check_cancel(progress_arr)
    kp = spec.get('kpm') or {}
    dm = spec.get('dmrg') or {}      # DMRG 截断控制（不传则用下面的默认值）
    res = eq_route.solve_equilibrium(
        system, want=want, betas=spec.get('betas'), backend=spec.get('method', 'auto'),
        num=kp.get('num', 400), nvec=kp.get('nvec', 1500), seed=kp.get('seed', 3),
        chi_max=int(dm.get('chi_max', 200)),
        truncerr=float(dm.get('truncerr', 1e-10)),
        max_sweeps=int(dm.get('max_sweeps', 60)),
        qmc_sweeps=int((spec.get('qmc') or {}).get('sweeps', 2000)),
        qmc_burn_in=int((spec.get('qmc') or {}).get('burn_in', 500)),
        qmc_measure_every=int((spec.get('qmc') or {}).get('measure_every', 5)),
        qmc_M=(spec.get('qmc') or {}).get('M'),
        memory_budget_mb=spec.get('memory_budget_mb'), dtype=spec.get('dtype', 'float64'),
        return_state=bool(spec.get('return_state', False)),
        exact=bool(spec.get('exact', want == 'gs')))
    check_cancel(progress_arr)
    res['method'] = want
    res['states'] = spec.get('states')
    res['labels'] = spec.get('labels')
    res['n_atoms'] = n
    errors = dict(res.get('error_report') or {})
    errors['warnings'] = list(errors.get('warnings') or []) + list(spec.get('warnings') or [])
    res['error_report'] = errors
    log('[多体平衡态] want=%s → 后端 %s，n=%d' % (want, res.get('backend'), n))
    report_progress(progress_arr, 1.0)
    return res


def solve_equilibrium(cache, params, want='gs', method=None, betas=None):
    """产品/AI 直接入口：快照 → 平衡态 RunSpec → Result。"""
    return run_equilibrium(build_equilibrium(cache, params, want=want, method=method,
                                             betas=betas))
