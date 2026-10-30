"""Backend names, execution entries and supported RunSpec options.

This module has no solver imports. Schema, GUI and execution share these
declarations; model-dependent restrictions remain in the numerical backend.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeBackend:
    module: str
    function: str
    aliases: tuple = ()
    options: tuple = ()  # (manybody suffix, engine keyword)
    segments: bool = False
    projected: bool = False
    gamma: bool = False
    return_state: bool = False
    charge: bool = False
    native_state: str = ''
    algorithm: str = ''


STEP = (('max_step_ns', 'max_step_ns'),)
TRUNCATION = (('chi_max', 'chi_max'), ('truncerr', 'truncerr'))
KRYLOV = (('krylov_dim', 'krylov_dim'), ('krylov_tol', 'krylov_tol'))
CPU_MPS = TRUNCATION + STEP + KRYLOV + (('max_sweeps', 'max_sweeps'),)
GPU_MPS = TRUNCATION + STEP + (('dtype', 'dtype'), ('device', 'device'),
                             ('svd_driver', 'svd_driver'),
                             ('memory_budget_mb', 'memory_budget_mb'))

TIME_BACKENDS = {
    'ed': TimeBackend('engine_ed', 'evolve_manybody',
                      options=(('memory_budget_mb', 'memory_budget_mb'),), segments=True,
                      projected=True, gamma=True, return_state=True),
    'tebd': TimeBackend('engine_mps', 'evolve_mps', options=CPU_MPS,
                        segments=True, return_state=True, charge=True, native_state='tenpy'),
    'tebd-gpu': TimeBackend('engine_mps_gpu', 'evolve_mps_gpu',
                            aliases=('mps-gpu', 'cuda-tebd', 'tebd-cuda'), options=GPU_MPS,
                            segments=True, return_state=True, native_state='torch', algorithm='tebd'),
    'swap-tebd-gpu': TimeBackend('engine_mps_gpu', 'evolve_mps_gpu',
                                 aliases=('swap-tebd', 'tebd-gpu-swap', 'cuda-swap-tebd'),
                                 options=GPU_MPS, segments=True, return_state=True,
                                 native_state='torch', algorithm='swap-tebd'),
    'tdvp': TimeBackend('engine_mps', 'evolve_mps', aliases=('mps', 'mps_mpo'),
                        options=CPU_MPS, segments=True, return_state=True,
                        charge=True, native_state='tenpy'),
    'tdvp-single': TimeBackend('engine_mps', 'evolve_mps',
                               aliases=('tdvp-1site', 'tdvp1', 'single-site-tdvp'),
                               options=CPU_MPS, segments=True, return_state=True,
                               charge=True, native_state='tenpy'),
    'tdvp-adaptive': TimeBackend('engine_mps', 'evolve_mps',
                                 aliases=('tdvp-hybrid', 'adaptive-tdvp'),
                                 options=CPU_MPS + (
                                     ('tdvp_adaptive_switch_chi', 'adaptive_switch_chi'),
                                     ('tdvp_adaptive_min_steps', 'adaptive_min_steps')),
                                 segments=True, return_state=True, charge=True,
                                 native_state='tenpy'),
    'tdvp-trajectories': TimeBackend('engine_trajectories', 'evolve_trajectories',
                                     aliases=('trajectories', 'mcwf'), gamma=True,
                                     options=TRUNCATION + STEP + KRYLOV + (
                                         ('n_trajectories', 'n_trajectories'),
                                         ('collapse_ops', 'collapse_ops'), ('seed', 'seed'))),
    'classical-spin': TimeBackend('classical_spin', 'evolve_classical_spin',
                                  aliases=('classical', 'llg'),
                                  options=STEP + (('classical_damping', 'damping'),)),
    'peps-real-time': TimeBackend('engine_peps', 'evolve_peps', aliases=('peps',),
                                  options=TRUNCATION + STEP + (
                                      ('grid_shape', 'grid_shape'),
                                      ('memory_budget_mb', 'max_intermediate_mb'))),
}

EQ_BACKENDS = {
    'ed': ('gs', 'thermal', 'dos'),
    'gpu-batch': ('gs',), 'gpu-lanczos': ('gs',), 'lanczos': ('gs',),
    'kpm': ('thermal', 'dos'), 'gpu-kpm': ('thermal', 'dos'),
    'free-fermion': ('gs',), 'dmrg': ('gs',), 'qmc': ('gs', 'thermal'),
}


def time_backend_names():
    return ['auto', *TIME_BACKENDS]


def equilibrium_backend_names():
    return ['auto', *EQ_BACKENDS]


def normalize_time_backend(name):
    name = str(name or 'auto').lower()
    if name == 'auto' or name in TIME_BACKENDS:
        return name
    for canonical, backend in TIME_BACKENDS.items():
        if name in backend.aliases:
            return canonical
    raise ValueError('未知多体 solver=%r；合法：%s' % (name, time_backend_names()))


def normalize_equilibrium_backend(name, want):
    name = str(name or 'auto').lower()
    name = {'kpm-gpu': 'gpu-kpm', 'free_fermion': 'free-fermion'}.get(name, name)
    if name != 'auto' and want not in EQ_BACKENDS.get(name, ()):
        raise ValueError('平衡态后端 %r 不支持 want=%r' % (name, want))
    return name


def engine_entries():
    return {(name, 'many'): ('.manybody.' + backend.module, backend.function)
            for name, backend in TIME_BACKENDS.items()}


def validate_time_capabilities(name, *, segments=None, plan=None, gamma=None,
                               return_state=False, charge_sector=None, psi0=None,
                               collapse_ops=None):
    import numpy as np
    backend = TIME_BACKENDS[normalize_time_backend(name)]
    if segments and not backend.segments:
        raise ValueError('%s 尚不支持 drive_segments' % name)
    if plan and plan.get('projected') and not backend.projected:
        raise ValueError('%s 尚不支持 projected HamiltonianPlan；请选择 ed' % name)
    if gamma is not None and np.any(np.asarray(gamma) != 0) and not backend.gamma:
        raise ValueError('%s 仅支持 gamma=0；耗散请使用 ED 或 trajectories' % name)
    if return_state and not backend.return_state:
        raise ValueError('%s 尚不支持 return_state' % name)
    if charge_sector is not None and not backend.charge:
        raise ValueError('%s 尚不支持实时 charge_sector；请选择 CPU MPS' % name)
    if collapse_ops and name != 'tdvp-trajectories':
        raise ValueError('collapse_ops 需要 tdvp-trajectories')
    if isinstance(psi0, str) and psi0 == 'ground' and backend.module not in ('engine_ed', 'engine_mps'):
        raise ValueError('%s 尚不支持 ground 初态；请先准备兼容初态' % name)
    if psi0 is not None and not isinstance(psi0, (str, dict, int, np.integer)):
        # Entangled vectors/superpositions are ED inputs. Product arrays have
        # a separate local-state grammar and remain cheap for every backend.
        is_superposition = (isinstance(psi0, (list, tuple)) and len(psi0)
                            and isinstance(psi0[0], str) and psi0[0] == 'sup')
        is_vector = isinstance(psi0, np.ndarray) and psi0.ndim == 1
        if backend.module != 'engine_ed' and (is_superposition or is_vector):
            raise ValueError('%s 不接收全空间纠缠向量/叠加式；请使用原生张量态' % name)
    if isinstance(psi0, dict):
        kind = psi0.get('kind')
        if kind == 'mps' and (not backend.native_state or
                             psi0.get('backend') != backend.native_state):
            raise ValueError('MPS 初态需要匹配的 %s 后端' % psi0.get('backend'))
        if kind in ('peps', 'tensor_network') and name != 'peps-real-time':
            raise ValueError('PEPS 初态需要 peps-real-time')
