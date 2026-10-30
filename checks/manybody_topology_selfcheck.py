# -*- coding: utf-8 -*-
"""模型无关 Fukui Chern 与闭环 flux 的回归。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core.manybody.topology import (fukui_chern, loop_flux, twist_pairs,
                                         twist_calibration,
                                         wilson_loop, spectral_flow,
                                         entanglement_spectrum,
                                         topology_result, topology_report)
from src.core.manybody.assemble import geometry
from src.core.manybody.model import ManyBodyModel


sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.diag([1.0, -1.0]).astype(complex)


def qzw(tx, ty):
    m = -1.0
    h = np.sin(tx) * sx + np.sin(ty) * sy + (m + np.cos(tx) + np.cos(ty)) * sz
    return np.linalg.eigh(h)[1][:, 0]


r = fukui_chern(qzw, grid=(12, 12))
assert abs(abs(r['chern_rounded']) - 1) == 0, r
assert r['min_link_overlap'] > 0.9, r
flux = loop_flux([
    {'i': 0, 'j': 1, 'J': np.exp(0.2j)},
    {'i': 1, 'j': 2, 'J': np.exp(0.3j)},
    {'i': 0, 'j': 2, 'J': np.exp(-0.1j)},
], [0, 1, 2])
assert abs(flux['phase_rad'] - 0.6) < 1e-12, flux
print('PASS topology primitives; Chern=%+.6f, loop phase=%.6f'
      % (r['chern'], flux['phase_rad']))

tw = twist_pairs([{'i': 0, 'j': 1, 'J': 1.0 + 0.0j}],
                 [[0.9, 0.2], [0.1, 0.2]], [1.0, 1.0], theta=[0.3, 0.4])
assert tw['windings'] == [[-1, 0]], tw
assert abs(tw['pairs'][0]['J'] - np.exp(0.3j)) < 1e-12, tw
tc = twist_calibration({'pairs': [{'i': 0, 'j': 1, 'J': 1.0 + 0.0j}],
                        'terms': ['I2']}, [[0.9, 0.2], [0.1, 0.2]],
                       [1.0, 1.0], theta=[0.3, 0.4])
assert abs(tc['pairs'][0]['J'] - np.exp(0.3j)) < 1e-12

wl = wilson_loop(lambda _: np.eye(2, dtype=complex), [0.0, 1.0, 2.0])
assert np.allclose(wl['wilson_matrix'], np.eye(2)), wl
assert np.allclose(wl['eigenphases_rad'], 0.0), wl


def sf(t):
    c, s = np.cos(t), np.sin(t)
    u = np.array([[c, -s], [s, c]], complex)
    return {'evals': np.array([t, t + 1.0]), 'vecs': u}


flow = spectral_flow(sf, [0.0, 0.3, 0.8])
assert flow['tracking'] == 'overlap' and np.allclose(flow['ground_gap'], 1.0), flow

bell = np.array([1.0, 0.0, 0.0, 1.0], complex) / np.sqrt(2.0)
es = entanglement_spectrum(bell, 2, 1)
assert es['rank'] == 2 and abs(es['entropy'] - np.log(2.0)) < 1e-12, es

report = topology_report([
    topology_result('chern', 'ok', data={'chern': -1}),
    topology_result('spectral_flow', 'finite_size_candidate'),
])
assert report['status'] == 'inconclusive'
print('PASS twist/Wilson/spectral-flow/entanglement/contract primitives')

R, th, ph, winding = geometry([0.9, 0.2, 0.0], [0.1, 0.2, 0.0],
                              periods_um=[1.0, 1.0], return_winding=True)
assert abs(R - 0.2) < 1e-12 and winding == [-1, 0], (R, winding)
m = ManyBodyModel(template='xy', terms=('I2',), periods_um=[8.0, 9.0])
assert ManyBodyModel.from_dict(m.to_dict()).periods_um == (8.0, 9.0)
print('PASS periodic geometry metadata/minimum-image roundtrip')
