# -*- coding: utf-8 -*-
"""MPO compression experiments for the dense TDVP prototype.

Two-site SVD sweeps over the MPO bond dimension; exact when no singular value
is discarded.  The goal is to see how far the long-range MPO can be compressed
before the TDVP pops are affected.
"""
import string
import numpy as np


def compress_mpo(W, chi_max, n_sweeps=3, trunc_cut=0.0):
    W = [w.copy() for w in W]
    n = len(W)
    discarded = []
    for _ in range(int(n_sweeps)):
        # right-moving: absorb S into the right tensor
        for i in range(n - 1):
            DL, DM, d, _ = W[i].shape
            DM2, DR, _, _ = W[i + 1].shape
            if DM != DM2:
                raise ValueError('MPO bond mismatch')
            A = np.tensordot(W[i], W[i + 1], axes=([1], [0]))
            mat = A.reshape(DL * d * d, DR * d * d)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            keep = int(np.sum(S > max(trunc_cut * S[0], 1e-15)))
            keep = max(1, min(int(chi_max), keep, len(S)))
            if keep < len(S):
                discarded.append(float(np.sum(S[keep:] ** 2) / np.sum(S ** 2)))
            U = U[:, :keep] * S[:keep][None, :]
            Vh = Vh[:keep, :]
            W[i] = U.reshape(DL, d, d, keep).transpose(0, 3, 1, 2)
            W[i + 1] = Vh.reshape(keep, DR, d, d)
        # left-moving: absorb S into the left tensor
        for i in range(n - 2, -1, -1):
            DL, DM, d, _ = W[i].shape
            DM2, DR, _, _ = W[i + 1].shape
            A = np.tensordot(W[i], W[i + 1], axes=([1], [0]))
            mat = A.reshape(DL * d * d, DR * d * d)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            keep = int(np.sum(S > max(trunc_cut * S[0], 1e-15)))
            keep = max(1, min(int(chi_max), keep, len(S)))
            if keep < len(S):
                discarded.append(float(np.sum(S[keep:] ** 2) / np.sum(S ** 2)))
            U = U[:, :keep]
            Vh = Vh[:keep, :] * S[:keep][:, None]
            W[i] = U.reshape(DL, d, d, keep).transpose(0, 3, 1, 2)
            W[i + 1] = Vh.reshape(keep, DR, d, d)
    return W, discarded


def apply_mpo_dense(W, v):
    """Apply a dense MPO to a dense vector (tiny-N validation)."""
    n = len(W)
    d = W[0].shape[2]
    letters = string.ascii_letters
    if 3 * n + 1 > len(letters):
        raise ValueError('n too large for this helper')
    w = [letters[i] for i in range(n + 1)]
    p = [letters[n + 1 + i] for i in range(n)]
    q = [letters[2 * n + 1 + i] for i in range(n)]
    subs = [w[i] + w[i + 1] + p[i] + q[i] for i in range(n)]
    subs.append(''.join(q))
    out = ''.join(p)
    Hv = np.einsum(','.join(subs) + '->' + out, *W, np.asarray(v).reshape((d,) * n),
                   optimize='greedy')
    return Hv.reshape(-1)


def compress_mpo_position(W, chi_max, n_sweeps=2, trunc_cut=0.0):
    """Compress while keeping each original bond's upper bound (no padding)."""
    W = [w.copy() for w in W]
    n = len(W)
    dims = [w.shape[0] for w in W] + [W[-1].shape[1]]
    discarded = []
    for _ in range(int(n_sweeps)):
        for i in range(n - 1):
            DL, DM, d, _ = W[i].shape
            DM2, DR, _, _ = W[i + 1].shape
            A = np.tensordot(W[i], W[i + 1], axes=([1], [0]))
            mat = A.reshape(DL * d * d, DR * d * d)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            keep = int(np.sum(S > max(trunc_cut * S[0], 1e-15)))
            keep = max(1, min(int(chi_max), keep, dims[i + 1], len(S)))
            if keep < len(S):
                discarded.append(float(np.sum(S[keep:] ** 2) / np.sum(S ** 2)))
            U = U[:, :keep] * S[:keep][None, :]
            Vh = Vh[:keep, :]
            W[i] = U.reshape(DL, d, d, keep).transpose(0, 3, 1, 2)
            W[i + 1] = Vh.reshape(keep, DR, d, d)
        for i in range(n - 2, -1, -1):
            DL, DM, d, _ = W[i].shape
            DM2, DR, _, _ = W[i + 1].shape
            A = np.tensordot(W[i], W[i + 1], axes=([1], [0]))
            mat = A.reshape(DL * d * d, DR * d * d)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            keep = int(np.sum(S > max(trunc_cut * S[0], 1e-15)))
            keep = max(1, min(int(chi_max), keep, dims[i + 1], len(S)))
            if keep < len(S):
                discarded.append(float(np.sum(S[keep:] ** 2) / np.sum(S ** 2)))
            U = U[:, :keep]
            Vh = Vh[:keep, :] * S[:keep][:, None]
            W[i] = U.reshape(DL, d, d, keep).transpose(0, 3, 1, 2)
            W[i + 1] = Vh.reshape(keep, DR, d, d)
    return W, discarded
