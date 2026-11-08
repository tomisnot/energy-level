"""Backend names, execution entries and supported RunSpec options.

This module has no solver imports. Schema, GUI and execution share these
declarations; model-dependent restrictions remain in the numerical backend.
"""
from dataclasses import dataclass


class ConfigConflict(ValueError):
    """A configuration key/combination is refused by the selected backend.

    ``keys`` is the machine-readable conflict list; the MCP layer maps this
    to ``kind='config_conflict'`` instead of wrapping an internal error.
    """

    def __init__(self, message, *, keys=()):
        super().__init__(message)
        self.keys = tuple(keys)


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
    order: bool = False
    #: Pauli 串能力：''=不支持；'full'=全空间/受限基态矢；'mps'=TeNPy MPS；
    #: 'torch'=CUDA 张量 MPS；'ensemble'=轨迹系综；'factorized'=经典因子化。
    pauli: str = ''
    native_state: str = ''
    algorithm: str = ''


STEP = (('max_step_ns', 'max_step_ns'),)
TRUNCATION = (('chi_max', 'chi_max'), ('truncerr', 'truncerr'))
KRYLOV = (('krylov_dim', 'krylov_dim'), ('krylov_tol', 'krylov_tol'))
CPU_MPS = TRUNCATION + STEP + KRYLOV + (('max_sweeps', 'max_sweeps'), ('mps_order','mps_order'))
GPU_MPS = TRUNCATION + STEP + (('dtype', 'dtype'), ('device', 'device'),
                             ('svd_driver', 'svd_driver'),
                             ('memory_budget_mb', 'memory_budget_mb'))

TIME_BACKENDS = {
    'ed': TimeBackend('engine_ed', 'evolve_manybody',
                      options=(('memory_budget_mb', 'memory_budget_mb'),), segments=True,
                      projected=True, gamma=True, return_state=True, pauli='full'),
    'tebd': TimeBackend('engine_mps', 'evolve_mps', options=CPU_MPS,
                        segments=True, return_state=True, charge=True,
                        pauli='mps', native_state='tenpy', order=True),
    'tebd-gpu': TimeBackend('engine_mps_gpu', 'evolve_mps_gpu',
                            aliases=('mps-gpu', 'cuda-tebd', 'tebd-cuda'), options=GPU_MPS,
                            segments=True, return_state=True, pauli='torch',
                            native_state='torch', algorithm='tebd'),
    'swap-tebd-gpu': TimeBackend('engine_mps_gpu', 'evolve_mps_gpu',
                                 aliases=('swap-tebd', 'tebd-gpu-swap', 'cuda-swap-tebd'),
                                 options=GPU_MPS, segments=True, return_state=True,
                                 pauli='torch',
                                 native_state='torch', algorithm='swap-tebd'),
    'tdvp-mpo': TimeBackend('mpo_tdvp', 'evolve_mpo_tdvp', options=CPU_MPS,
                            segments=False, return_state=False, pauli=''),
    'tdvp-dense': TimeBackend('tdvp_dense', 'evolve_tdvp_dense',
                              options=TRUNCATION + STEP + KRYLOV + (
                                  ('mps_order', 'mps_order'),),
                              segments=True, projected=False, gamma=False,
                              return_state=False, charge=False, pauli='mps',
                              order=True, algorithm='tdvp-dense'),
    'tdvp': TimeBackend('engine_mps', 'evolve_mps', aliases=('mps', 'mps_mpo'),
                        options=CPU_MPS, segments=True, return_state=True,
                        charge=True, pauli='mps', native_state='tenpy', order=True),
    'tdvp-single': TimeBackend('engine_mps', 'evolve_mps',
                               aliases=('tdvp-1site', 'tdvp1', 'single-site-tdvp'),
                               options=CPU_MPS, segments=True, return_state=True,
                               charge=True, pauli='mps',
                               native_state='tenpy', order=True),
    'tdvp-adaptive': TimeBackend('engine_mps', 'evolve_mps',
                                 aliases=('tdvp-hybrid', 'adaptive-tdvp'),
                                 options=CPU_MPS + (
                                     ('tdvp_adaptive_switch_chi', 'adaptive_switch_chi'),
                                     ('tdvp_adaptive_min_steps', 'adaptive_min_steps')),
                                 segments=True, return_state=True, charge=True,
                                 pauli='mps', native_state='tenpy', order=True),
    'tdvp-trajectories': TimeBackend('engine_trajectories', 'evolve_trajectories',
                                     aliases=('trajectories', 'mcwf'), gamma=True,
                                     options=TRUNCATION + STEP + KRYLOV + (
                                         ('n_trajectories', 'n_trajectories'),
                                         ('collapse_ops', 'collapse_ops'), ('seed', 'seed')),
                                     pauli='ensemble'),
    'classical-spin': TimeBackend('classical_spin', 'evolve_classical_spin',
                                  aliases=('classical', 'llg'),
                                  options=STEP + (('classical_damping', 'damping'),),
                                  pauli='factorized'),
    'peps-real-time': TimeBackend('engine_peps', 'evolve_peps', aliases=('peps',),
                                  options=TRUNCATION + STEP + (
                                      ('grid_shape', 'grid_shape'),
                                      ('memory_budget_mb', 'max_intermediate_mb'))),
}

EQ_BACKENDS = {
    'ed': ('gs', 'thermal', 'dos', 'spectrum'),
    'idmrg': ('gs',),
    'gpu-batch': ('gs',), 'gpu-lanczos': ('gs',), 'lanczos': ('gs', 'spectrum'),
    'kpm': ('thermal', 'dos'), 'gpu-kpm': ('thermal', 'dos'),
    'free-fermion': ('gs',), 'dmrg': ('gs',), 'qmc': ('gs', 'thermal'),
}

# Optional finite-MPS facilities, separate from task/backend selection.
# Public dispatch checks this declaration before forwarding these options.
MPS_DTYPES = ('auto', 'real', 'complex')
EQ_FEATURES = {'dmrg': frozenset(('mps_dtype', 'mps_analysis', 'mps_order', 'mps_initial_state'))}


def require_equilibrium_features(backend, features):
    unsupported=set(features)-EQ_FEATURES.get(backend,frozenset())
    if unsupported:
        raise ConfigConflict(
            '%s does not support equilibrium options: %s' %
            (backend, ', '.join(sorted(unsupported))),
            keys=tuple('manybody.' + key for key in sorted(unsupported)))


def time_backend_names():
    return ['auto', *TIME_BACKENDS]


def backend(name):
    """规范名 → TimeBackend（未知名报错，不返回 None）。"""
    return TIME_BACKENDS[normalize_time_backend(name)]


def pauli_backends():
    """能给出纯态 Pauli 串的含时后端（按能力声明，不手抄名单）。"""
    return [name for name, b in TIME_BACKENDS.items()
            if b.pauli in ('full', 'mps', 'torch')]


def equilibrium_pauli_backends():
    """平衡态 gs 里能算 Pauli 串的后端。"""
    return ['ed', 'lanczos', 'gpu-lanczos', 'dmrg']


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
                               collapse_ops=None, mps_order=None, pauli_strings=None):
    import numpy as np
    backend = TIME_BACKENDS[normalize_time_backend(name)]
    if pauli_strings and not backend.pauli:
        alts = ' / '.join(pauli_backends())
        raise ConfigConflict(
            '%s 不提供可求 Pauli 串的多体态（能力=%r）；请改用 %s。'
            '该键不会被静默忽略' % (name, backend.pauli or 'none', alts),
            keys=('manybody.pauli_strings',))
    if mps_order is not None and not backend.order:
        raise ConfigConflict('%s does not support explicit mps_order' % name,
                                    keys=('manybody.mps_order',))
    if segments and not backend.segments:
        raise ConfigConflict('%s 尚不支持 drive_segments' % name,
                                    keys=('manybody.drive_segments',))
    if plan and plan.get('projected') and not backend.projected:
        raise ConfigConflict('%s 尚不支持 projected HamiltonianPlan；请选择 ed' % name,
                                    keys=('manybody.constraint',))
    if gamma is not None and np.any(np.asarray(gamma) != 0) and not backend.gamma:
        raise ConfigConflict('%s 仅支持 gamma=0；耗散请使用 ED 或 trajectories' % name,
                                    keys=('manybody.gamma_mhz',))
    if return_state and not backend.return_state:
        raise ConfigConflict('%s 尚不支持 return_state' % name,
                                    keys=('manybody.return_state',))
    if charge_sector is not None and not backend.charge:
        raise ConfigConflict('%s 尚不支持实时 charge_sector；请选择 CPU MPS' % name,
                                    keys=('manybody.charge_sector',))
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
