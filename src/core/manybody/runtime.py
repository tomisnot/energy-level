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


def reject_options(options):
    if options:
        raise TypeError('不支持的多体引擎选项：%s' % ', '.join(sorted(options)))


def segment_intervals(a, b, segments):
    """Split an output interval at every piecewise-constant drive boundary."""
    boundaries = [float(a)] + sorted({float(s['t0']) for s in segments or []
                                     if a < float(s['t0']) < b}) + [float(b)]
    return zip(boundaries[:-1], boundaries[1:])


def segment_drive(t, segments, n):
    """Existing segment contract: absent Om/delta/phase components mean zero."""
    if not segments:
        return None
    segment = sorted(segments, key=lambda s: float(s['t0']))[0]
    for s in sorted(segments, key=lambda s: float(s['t0'])):
        if float(s['t0']) <= t:
            segment = s
    om = np.asarray(segment.get('Om', np.zeros(n)), float)
    phase = np.asarray(segment.get('phase', np.zeros(n)), float)
    return om * np.exp(1j * phase), np.asarray(segment.get('delta', np.zeros(n)), float)
