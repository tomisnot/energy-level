"""Versioned data semantics shared by direct solver and RunSpec consumers.

Legacy arrays remain available. Descriptors refer to their paths and do not
copy numerical buffers; consumers can plot by quantity/axes instead of solver.
"""
from functools import wraps
from inspect import signature
import numpy as np


def describe_result(result, calib=None, *, loss_kind='norm_drift', approximation=None):
    out = result
    n = len(calib['atoms']) if calib else int(out.get('n_atoms', out.get('n_sites', 0)))
    descriptors = {}

    def add(path, quantity, axes, unit='1', **extra):
        value = out
        for key in path.split('.'):
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            array = np.asarray(value)
            if array.size == 0:
                return
            descriptors[path] = dict(quantity=quantity, axes=list(axes), unit=unit, **extra)
            if (path in ('S', 'gap', 'observables.phase')
                    and np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()):
                descriptors[path]['missing_values'] = 'nonfinite'

    if out.get('pops') is not None:
        p = np.asarray(out['pops'])
        labels = out.get('states') or []
        if p.ndim != 2 or p.shape[1] != len(labels):
            raise ValueError('多体结果契约：pops 每列必须有对应 states 身份')
        kind = ('basis' if labels and 'index' in labels[0] else 'site')
        out['pops_kind'] = 'basis_probability' if kind == 'basis' else 'site_occupation'
        add('pops', out['pops_kind'], ('time', kind),
            summary=dict(mode='top', n=8, labels='states'))
    for key in ('pops_atom', 'occupations'):
        add(key, 'site_occupation', ('time', 'site'))
    add('basis_probabilities', 'basis_probability', ('time', 'basis'))
    add('stderr', 'standard_error', ('time', 'site'), of='pops_atom')
    add('S', 'entanglement_entropy', ('time',) if out.get('t') is not None else (), 'nat',
        **({'summary': dict(mode='scalar')} if out.get('t') is None else {}))
    corr_3d = (out.get('betas') is not None and out.get('corr') is not None
               and np.asarray(out['corr']).ndim == 3)
    add('corr', 'connected_occupation_correlation',
        ('beta', 'site', 'site') if corr_3d else ('site', 'site'),
        **({'summary': dict(mode='first_last_axis', axis=0)} if corr_3d else {}))
    add('energy', 'energy', ('time',), 'MHz')
    add('E0', 'ground_state_energy', (), 'MHz', summary=dict(mode='scalar'))
    add('E_per_site', 'energy_per_site', (), 'MHz')
    add('E_per_cell', 'energy_per_cell', (), 'MHz')
    add('S_per_bond', 'entanglement_entropy_per_bond', (), 'nat')
    add('entanglement_entropy', 'entanglement_entropy', ('site',), 'nat')
    add('corr_length', 'correlation_length', (), 'site')
    add('chi', 'mps_bond_dimension', (), '1', summary=dict(mode='scalar'))
    add('trunc_err', 'truncation_error', (), '1')
    add('unit_cell_size', 'unit_cell_size', (), '1')
    add('energy_expectation', 'energy_expectation', (), 'MHz')
    add('energy_variance', 'energy_residual_variance', (), 'MHz^2')
    add('permutation_overlap', 'site_permutation_expectation', (), '1')
    add('projection_weight', 'symmetry_projection_weight', (), '1')
    add('gap', 'energy_gap', (), 'MHz', summary=dict(mode='scalar'))
    add('evals', 'eigenenergy', ('eigenstate',), 'MHz')
    for key, quantity, unit in (('E', 'energy', 'MHz'), ('C', 'heat_capacity', '1'),
                                 ('F', 'free_energy', 'MHz')):
        add(key, quantity, ('beta',), unit, summary=dict(mode='first_last'))
    add('E_err', 'standard_error', ('beta',) if out.get('betas') is not None else (), 'MHz', of='E')
    add('rho', 'density_of_states', ('energy',), '1/MHz')
    # 源态谱（want=spectrum）：横轴用 `energies`（频率网格），纵轴是谱函数。
    # 未展宽的求积权重留在结果里（程序化用），不声明成可绘制的 dataset——它的横轴
    # 是 Ritz 节点，与 `energies` 是两条不同的坐标轴。
    add('spectral', 'spectral_function', ('energy',), '1/MHz')
    add('source_norm', 'source_state_weight', (), '1')
    for key, quantity, unit in (('gaps','energy_gap','MHz'),
                                 ('ns','occupation_density','1'),
                                 ('Om_t','drive_amplitude','MHz'),
                                 ('de_t','detuning','MHz')):
        add(key, quantity, ('tau',), unit)
    if out.get('t') is not None:
        n_atom_summary = {}
    elif out.get('betas') is not None:
        n_atom_summary = {'summary': dict(mode='rows', n=8)}
    else:
        n_atom_summary = {'summary': dict(mode='list')}
    add('n_atom', 'site_occupation', ('time','site') if out.get('t') is not None else
        ('beta', 'site') if out.get('betas') is not None else ('site',),
        **n_atom_summary)
    for key in ('sx', 'sy', 'sz', 'sp', 'sm', 'phase'):
        add('observables.'+key, key, ('time', 'site'), 'rad' if key == 'phase' else '1')
    for key, axes, unit in (('xy_corr', ('site', 'site'), '1'),
                             ('link_corr', ('time', 'link'), '1'),
                             ('loop_phase_series', ('time', 'loop'), 'rad'),
                             ('loop_winding_series', ('time', 'loop'), '1')):
        add('observables.'+key, key, axes, unit)
    # 任意阶 Pauli 字符串：含时是 (time, string)，平衡态是 (string,)（同一条字符串
    # 在两种请求下语义相同，只是含时多一个时间轴）。
    time_pauli = out.get('t') is not None
    add('observables.pauli_strings', 'pauli_string_expectation',
        ('time', 'string') if time_pauli else ('string',), '1',
        summary=dict(mode='final' if time_pauli else 'head', n=12,
                     labels=('observables.pauli_string_labels' if time_pauli
                             else 'pauli_string_labels')),
        **({'items': dict(labels='observables.pauli_string_labels')}
           if time_pauli else {}))
    add('pauli_strings', 'pauli_string_expectation', ('string',), '1',
        summary=dict(mode='head', n=12, labels='pauli_string_labels'))
    add('observables.loop_phase', 'loop_phase', ('loop',), 'rad')
    add('observables.loop_winding', 'loop_winding', ('loop',))
    if out.get('loss') is not None:
        # Compatibility alias: its explicit physical/numerical meaning travels
        # with the data, rather than being inferred from a backend name.
        out[loss_kind] = out['loss']
        add(loss_kind, loss_kind, ('time',))
        add('loss', loss_kind, ('time',), alias_of=loss_kind)
    add('survival_weight', 'no_jump_survival_weight', ('time',))
    add('jump_counts', 'jump_count', ('trajectory',))
    add('spins', 'classical_spin', ('time', 'site', 'component'))
    out['data_schema'] = dict(version=1, quantities=descriptors,
                              axes=dict(time=dict(field='t', unit='ns'),
                                        beta=dict(field='betas', unit='1/MHz'),
                                        energy=dict(field='energies', unit='MHz'),
                                        tau=dict(field='taus', unit='1'),
                                        site=dict(field='sites', size=n),
                                        link=dict(field='observables.link_pairs'),
                                        loop=dict(field='loops'),
                                        trajectory=dict(field='jump_counts'),
                                        eigenstate=dict(field='evals'),
                                        component=dict(values=['x','y','z']),
                                        basis=dict(field=('basis_states' if out.get('basis_states') is not None
                                                          else 'states'), bit_order='site0-msb')),
                              approximation=approximation)
    if calib:
        out['sites'] = [dict(site_id=a.get('id', i),
                             **({'pos':list(a['pos'])} if 'pos' in a else {}))
                        for i, a in enumerate(calib['atoms'])]
    return out


def _at_path(data, path):
    node = data
    for key in str(path).split('.'):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _json_num(value):
    z = complex(value)
    if z.imag == 0.0:
        return round(float(z.real), 9)
    return [round(float(z.real), 9), round(float(z.imag), 9)]


def _json_array(value):
    a = np.asarray(value)
    if a.ndim == 0:
        return _json_num(a)
    return [_json_array(x) for x in a]


def _labels_at(result, path):
    if not path:
        return []
    value = _at_path(result, path)
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        if isinstance(item, dict):
            out.append(str(item.get('label', item.get('site_id', ''))))
        else:
            out.append(str(item))
    return out


def _label(labels, i):
    return labels[i] if i < len(labels) else '#%d' % i


def summarize_result(result, *, labels=None, top_n=None):
    """Descriptor-driven compact summary (``data_schema.quantities[*].summary``).

    The result contract declares *what* is small enough to summarize; this
    function only applies the policy.  No quantity-specific branches: Pauli,
    equilibrium scalars and future observables use the same modes.
    """
    schema = result.get('data_schema') or {}
    quantities = schema.get('quantities') or {}
    out = {}
    for path, meta in quantities.items():
        policy = meta.get('summary')
        if not policy:
            continue
        value = _at_path(result, path)
        if value is None:
            continue
        a = np.asarray(value)
        if a.size == 0:
            continue
        mode = policy.get('mode')
        labs = (labels if path == 'pops' and labels is not None
                else _labels_at(result, policy.get('labels')))
        if mode == 'scalar':
            if a.ndim == 0 or a.size == 1:
                out[path] = dict(mode='scalar',
                                 value=_json_num(a.reshape(-1)[0]))
        elif mode == 'list':
            vals = a.reshape(-1)
            out[path] = dict(mode='list',
                             values=[_json_num(x) for x in vals],
                             total=int(vals.size))
        elif mode == 'head':
            vals = a.reshape(-1)
            n = policy.get('n')
            k = vals.size if n is None else min(int(n), vals.size)
            out[path] = dict(mode='head',
                             items={_label(labs, i): _json_num(vals[i])
                                    for i in range(k)},
                             total=int(vals.size))
        elif mode == 'final':
            if a.ndim == 2:
                vals, total = a[-1], int(a.shape[1])
            else:
                vals, total = a.reshape(-1), int(a.size)
            k = vals.size if policy.get('n') is None else min(int(policy['n']), vals.size)
            out[path] = dict(mode='final',
                             items={_label(labs, i): _json_num(vals[i])
                                    for i in range(k)},
                             total=total)
        elif mode == 'top':
            if a.ndim == 2:
                avg, final, total = a.mean(axis=0), a[-1], int(a.shape[1])
            else:
                avg = final = a.reshape(-1)
                total = int(avg.size)
            nn = (top_n if path == 'pops' and top_n is not None
                  else policy.get('n'))
            k = min(int(nn or 8), avg.size)
            order = np.argsort(-avg)[:k]
            out[path] = dict(
                mode='top',
                top={_label(labs, int(i)): _json_num(avg[i]) for i in order},
                final={_label(labs, int(i)): _json_num(final[i]) for i in order},
                total=total)
        elif mode == 'rows':
            if a.ndim == 2:
                n = policy.get('n')
                k = a.shape[0] if n is None else min(int(n), a.shape[0])
                out[path] = dict(mode='rows',
                                 rows=[_json_array(a[i]) for i in range(k)],
                                 total=int(a.shape[0]))
        elif mode == 'first_last_axis':
            axis = int(policy.get('axis', 0))
            if a.ndim > axis:
                first = np.take(a, 0, axis=axis)
                last = np.take(a, -1, axis=axis)
                out[path] = dict(mode='first_last_axis',
                                 first=_json_array(first),
                                 last=_json_array(last),
                                 total=int(a.shape[axis]))
        elif mode == 'first_last':
            vals = a.reshape(-1)
            if vals.size:
                out[path] = dict(mode='first_last',
                                 first=_json_num(vals[0]),
                                 last=_json_num(vals[-1]),
                                 total=int(vals.size))
    return out


def time_result(*, loss_kind='norm_drift', approximation=None):
    def decorate(fn):
        call_signature = signature(fn)
        @wraps(fn)
        def run(*args, **kwargs):
            result = fn(*args, **kwargs)
            calib = args[0] if args else kwargs['calib']
            loops = call_signature.bind(*args, **kwargs).arguments.get('loops')
            if loops is not None:
                result['loops'] = [list(loop) for loop in loops]
            kind = loss_kind(result) if callable(loss_kind) else loss_kind
            return describe_result(result, calib, loss_kind=kind, approximation=approximation)
        return run
    return decorate
