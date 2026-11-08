"""Shared time-grid validation and cooperative progress/cancellation."""
import numpy as np


def time_grid(t):
    t = np.asarray(t, float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) < 0):
        raise ValueError('t_grid 必须包含至少两个有限、单调不减的时间点')
    return t


def check_cancel(progress_arr):
    if progress_arr is not None and float(progress_arr[1]) != 0:
        raise RuntimeError('用户取消')


def report_progress(progress_arr, fraction):
    check_cancel(progress_arr)
    if progress_arr is not None:
        progress_arr[0] = float(fraction)


def progress_range(progress_arr, start, stop):
    """Compose subtask progress while forwarding the shared cancel flag."""
    if not 0 <= start < stop <= 1:
        raise ValueError('Progress range must satisfy 0 <= start < stop <= 1')
    if progress_arr is None:
        return None
    class Range:
        def __getitem__(self,index):
            return ((progress_arr[0]-start)/(stop-start) if index==0 else progress_arr[index])
        def __setitem__(self,index,value):
            if index==0:progress_arr[0]=start+(stop-start)*float(value)
            else:progress_arr[index]=value
    return Range()


def reject_options(options):
    if options:
        raise TypeError('不支持的多体引擎选项：%s' % ', '.join(sorted(options)))


def atom_to_mps(values, order):
    """Reorder atom-indexed local values into the explicit MPS order."""
    return [values[int(a)] for a in order]


def mps_to_atom(values, order):
    """Reorder MPS-site values back to atom order (one declaration)."""
    vals = np.asarray(values)
    out = np.empty(len(order), dtype=vals.dtype)
    out[np.asarray(order, dtype=int)] = vals
    return out


def segment_intervals(a, b, segments):
    """Split an output interval at every piecewise-constant drive boundary."""
    boundaries = [float(a)] + sorted({float(s['t0']) for s in segments or []
                                     if a < float(s['t0']) < b}) + [float(b)]
    return zip(boundaries[:-1], boundaries[1:])


def _segment_arrays(segment, n):
    om = np.asarray(segment.get('Om', np.zeros(n)), float).reshape(-1)
    phase = np.asarray(segment.get('phase', np.zeros(n)), float).reshape(-1)
    de = np.asarray(segment.get('delta', np.zeros(n)), float).reshape(-1)
    if om.size != n or phase.size != n or de.size != n:
        raise ValueError('segment_drive: 每段的 Om/phase/delta 长度必须等于 n')
    return om * np.exp(1j * phase), de


def time_steps(a, b, segments=None, max_step_ns=None):
    """Internal step grid shared by MPS/dense time evolvers.

    Splits at piecewise-constant boundaries, then at ``max_step_ns`` inside
    each piece.  Keeping this in one place prevents the ED/MPS/dense paths
    from drifting on their step construction.
    """
    steps = []
    pieces = segment_intervals(a, b, segments) if segments else [(float(a), float(b))]
    for left, right in pieces:
        span = float(right) - float(left)
        nsub = (1 if max_step_ns is None or float(max_step_ns) <= 0 else
                max(1, int(np.ceil(span / float(max_step_ns)))))
        grid = np.linspace(float(left), float(right), nsub + 1)
        steps.extend(zip(grid[:-1], grid[1:]))
    return steps


def segment_drive(t, segments, n, *, interp='left'):
    """Resolve a piecewise-constant drive at time ``t``.

    ``interp='left'`` is the historical, bit-identical rule (latest ``t0``).
    ``interp='midpoint'`` treats the entries as control points of a linear
    ramp and returns the value at the **midpoint of the containing segment
    interval** (the average of its two endpoint values; the last interval uses
    its right endpoint).  This is a midpoint rule for a smooth ramp and keeps
    one model per segment, so the existing engine rebuild schedule is intact.
    """
    if not segments:
        return None
    if interp not in ('left', 'midpoint'):
        raise ValueError('drive_interp 须为 left/midpoint，得到 %r' % (interp,))
    ordered = sorted(segments, key=lambda s: float(s['t0']))
    index = 0
    for k, s in enumerate(ordered):
        if float(s['t0']) <= t:
            index = k
    om, de = _segment_arrays(ordered[index], n)
    if interp == 'midpoint' and index < len(ordered) - 1:
        om2, de2 = _segment_arrays(ordered[index + 1], n)
        om = 0.5 * (om + om2)
        de = 0.5 * (de + de2)
    return om, de


def _finite(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def describe_drive(segments, Om, delta, n, t_max, *, n_trace=160, max_traces=3,
                   interp='left'):
    """One definition of the **effective many-body drive** for web/MCP/card.

    Returns a bounded, JSON-facing view: static vs segmented, a compact segment
    table, and (optionally) a few real traces.  ``segment_drive`` remains the
    only place that knows how a segment resolves at a time.
    """
    n = int(n)
    Om = (np.zeros(n, complex) if Om is None
          else np.asarray(Om, complex).reshape(-1))
    delta = (np.zeros(n, float) if delta is None
             else np.asarray(delta, float).reshape(-1))
    if Om.size != n or delta.size != n:
        raise ValueError('describe_drive: Om/delta 长度必须等于 n')
    segments = list(segments or [])
    out = {'kind': 'segments' if segments else 'static',
           'n_segments': len(segments), 'interp': interp,
           'segments': [], 'traces': []}
    if segments:
        for s in sorted(segments, key=lambda x: float(x['t0'])):
            om = np.asarray(s.get('Om', np.zeros(n)), float).reshape(-1)
            de = np.asarray(s.get('delta', np.zeros(n)), float).reshape(-1)
            ph = np.asarray(s.get('phase', np.zeros(n)), float).reshape(-1)
            if om.size != n or de.size != n or ph.size != n:
                raise ValueError('describe_drive: segment 的 Om/delta/phase 长度必须等于 n')
            out['segments'].append(dict(
                t0=float(s['t0']),
                omega_max=float(np.max(np.abs(om))) if n else 0.0,
                delta_max=float(np.max(np.abs(de))) if n else 0.0,
                phase_max=float(np.max(np.abs(ph))) if n else 0.0))
    end = _finite(t_max)
    if end is None or end <= 0 or not n:
        return out
    t = np.linspace(0.0, end, int(max(2, n_trace)))
    drive = ([segment_drive(float(x), segments, n, interp=interp) for x in t]
             if segments else [None] * len(t))
    for site in range(min(n, int(max_traces))):
        out['traces'].append(dict(
            label='Ω · 原子 %d' % site,
            x=[_finite(x) for x in t],
            values=[_finite(abs(v[0][site]) if v is not None else abs(Om[site]))
                    for v in drive],
            unit='MHz'))
        if segments or np.any(delta):
            out['traces'].append(dict(
                label='δ · 原子 %d' % site,
                x=[_finite(x) for x in t],
                values=[_finite(v[1][site]) if v is not None else _finite(delta[site])
                        for v in drive],
                unit='MHz'))
    return out
