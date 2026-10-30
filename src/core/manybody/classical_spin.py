# -*- coding: utf-8 -*-
"""Classical Bloch-spin dynamics for large two-level arrays.

The input is the same calibrated point-cloud model as the quantum engines. A
site is a unit Bloch vector in the local (|g>,|r>) basis; this backend solves
the classical mean-field Hamiltonian dynamics, with optional dissipative
Landau--Lifshitz damping. It is a separate physical approximation, never an
implicit fallback for quantum solvers.
"""
from .runtime import time_grid, check_cancel, report_progress, reject_options
from .backends import validate_time_capabilities
from .results import time_result
import numpy as np
from ..evolve_util import CONV
from .state import product_vectors
from .observables import link_loop_observables
__all__ = ["spins_from_product", "energy", "effective_field", "evolve_classical_spin"]

def spins_from_product(spec, n):
    vecs = product_vectors(spec, n)
    if vecs is None: raise ValueError("classical-spin 不会隐式求量子 ground；请提供局部乘积态或 spins0")
    out = np.zeros((int(n),3), float)
    for i,a in enumerate(vecs):
        out[i] = (2*np.real(np.conj(a[0])*a[1]), 2*np.imag(np.conj(a[0])*a[1]), float(abs(a[0])**2-abs(a[1])**2))
    return out

def _arrays(calib, Om, delta):
    n = len(calib.get("atoms") or [])
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n, float) if delta is None else np.asarray(delta, float)
    if Om.shape != (n,) or delta.shape != (n,): raise ValueError("classical-spin 的 Om/delta 长度必须等于原子数")
    return Om,delta

def _pair_terms(calib, terms):
    use = set(("I1","I2","I3") if terms is None else terms)
    for p in calib.get("pairs") or []:
        yield use,int(p["i"]),int(p["j"]),complex(p.get("V",0)),complex(p.get("J",0)),complex(p.get("A",0))

def energy(spins, calib, Om=None, delta=None, terms=None):
    s=np.asarray(spins,float); n=len(calib.get("atoms") or [])
    if s.shape!=(n,3): raise ValueError("spins 形状必须是 (N,3)")
    Om,delta=_arrays(calib,Om,delta); e=float(np.sum(delta*(1-s[:,2])/2))
    e += float(np.sum((Om.real*s[:,0]+Om.imag*s[:,1])/2))
    for use,i,j,V,J,A in _pair_terms(calib,terms):
        if "I1" in use: e += float((V.real)*(1-s[i,2])*(1-s[j,2])/4)
        zi=s[i,0]-1j*s[i,1]; zj=s[j,0]+1j*s[j,1]
        if "I2" in use: e += float(0.5*np.real(J*zi*zj))
        if "I3" in use: e += float(0.5*np.real(A*zi*np.conj(zj)))
    return float(e)

def effective_field(spins, calib, Om=None, delta=None, terms=None):
    s=np.asarray(spins,float); n=len(calib.get("atoms") or [])
    if s.shape!=(n,3): raise ValueError("spins 形状必须是 (N,3)")
    Om,delta=_arrays(calib,Om,delta); g=np.zeros_like(s)
    g[:,0]+=Om.real/2; g[:,1]+=Om.imag/2; g[:,2]-=delta/2
    for use,i,j,V,J,A in _pair_terms(calib,terms):
        if "I1" in use:
            g[i,2]-=V.real*(1-s[j,2])/4; g[j,2]-=V.real*(1-s[i,2])/4
        # E_J=1/2 Re[J (xi-i yi)(xj+i yj)] and E_A=1/2 Re[A (...)(...)]
        if "I2" in use:
            q=J*(s[j,0]+1j*s[j,1]); g[i,0]+=0.5*q.real; g[i,1]+=0.5*q.imag
            q=np.conj(J)*(s[i,0]+1j*s[i,1]); g[j,0]+=0.5*q.real; g[j,1]+=0.5*q.imag
        if "I3" in use:
            q=A*(s[j,0]-1j*s[j,1]); g[i,0]+=0.5*q.real; g[i,1]+=0.5*q.imag
            q=A*(s[i,0]-1j*s[i,1]); g[j,0]+=0.5*q.real; g[j,1]+=0.5*q.imag
    return g

@time_result(approximation='factorized classical spin')
def evolve_classical_spin(calib, psi0=None, t_grid=None, Om=None, delta=None, gamma=None,
                          terms=None, damping=0.0, integrator="rk4", renormalize=True,
                          max_step_ns=0.05, spins0=None, want_observables=True,
                          link_pairs=None, loops=None, segments=None, plan=None,
                          progress_arr=None, log=None, **kwargs):
    reject_options(kwargs)
    validate_time_capabilities('classical-spin', segments=segments, plan=plan, gamma=gamma)
    t_grid = time_grid(t_grid)
    check_cancel(progress_arr)
    n=len(calib.get("atoms") or [])
    if gamma is not None and np.any(np.asarray(gamma,float) != 0): raise ValueError("classical-spin 不把量子衰减 gamma 静默当成经典动力学")
    spins=spins_from_product(psi0,n) if spins0 is None else np.asarray(spins0,float).copy()
    if spins.shape!=(n,3): raise ValueError("spins0 形状必须是 (N,3)")
    norms=np.linalg.norm(spins,axis=1)
    if np.any(norms<=0): raise ValueError("classical-spin 初态包含零向量")
    spins/=norms[:,None]; t=np.asarray(t_grid,float)
    if t.ndim!=1 or len(t)<2 or np.any(np.diff(t)<0): raise ValueError("classical-spin 时间网格必须单调且至少含两个点")
    if max_step_ns<=0: raise ValueError("max_step_ns 必须 >0")
    damp=float(damping)
    def rhs(x):
        g=effective_field(x,calib,Om,delta,terms)
        # spin-1/2 Poisson bracket: ds/dt = 2*CONV*(g x s).
        v=2*CONV*np.cross(g,x)
        if damp: v += 2*CONV*damp*np.cross(x,np.cross(x,g))
        return v
    states=np.zeros((len(t),n,3)); energies=np.zeros(len(t)); states[0]=spins; energies[0]=energy(spins,calib,Om,delta,terms)
    max_norm_dev=0.; max_step=0
    for k in range(1,len(t)):
        interval=float(t[k]-t[k-1]); ns=max(1,int(np.ceil(interval/max_step_ns))); h=interval/ns; max_step=max(max_step,h)
        for _ in range(ns):
            check_cancel(progress_arr)
            if str(integrator).lower()=="euler": spins += h*rhs(spins)
            else:
                k1=rhs(spins); k2=rhs(spins+.5*h*k1); k3=rhs(spins+.5*h*k2); k4=rhs(spins+h*k3); spins += h*(k1+2*k2+2*k3+k4)/6
            if renormalize: spins/=np.linalg.norm(spins,axis=1)[:,None]
        report_progress(progress_arr, k/(len(t)-1))
        states[k]=spins; energies[k]=energy(spins,calib,Om,delta,terms); max_norm_dev=max(max_norm_dev,float(np.max(np.abs(np.linalg.norm(spins,axis=1)-1))))
    pops=(1-states[:,:,2])/2
    obs = None
    if want_observables:
        sx, sy, sz = states[:, :, 0], states[:, :, 1], states[:, :, 2]
        sm = (sx + 1j * sy) / 2.0
        sp = np.conj(sm)
        phase = np.angle(sm); phase[np.abs(sm) < 1e-14] = np.nan
        link_series = None
        if link_pairs is not None:
            link_series = np.empty((len(t), len(link_pairs)), complex)
            for q, (a, b) in enumerate(link_pairs):
                a, b = int(a), int(b)
                link_series[:, q] = (pops[:, a] if a == b
                                     else sp[:, a] * sm[:, b])
        loop_phase_series, loop_winding_series = link_loop_observables(
            link_series, link_pairs, loops) if link_series is not None and loops is not None else (
                np.empty((0, 0), float), np.empty((0, 0), float))
        lp = loop_phase_series[-1].tolist() if loop_phase_series.size else []
        lw = loop_winding_series[-1].tolist() if loop_winding_series.size else []
        obs = dict(sx=sx, sy=sy, sz=sz, sp=sp, sm=sm, phase=phase,
                   xy_corr=None, loop_phase=lp, loop_winding=lw,
                   loop_phase_series=loop_phase_series,
                   loop_winding_series=loop_winding_series,
                   link_corr=link_series, link_pairs=link_pairs,
                   approximation='factorized classical-spin')
    return dict(t=t, spins=states, pops=pops, pops_atom=pops, n_atom=pops, states=[{"label":"atom%d"%i,"atom_id":i} for i in range(n)], method="manybody", backend="classical-spin", loss=np.zeros(len(t)), energy=energies,
                observables=obs,
                error_report=dict(backend="classical-spin", damping=damp, integrator=integrator, internal_step_ns=max_step, max_spin_norm_dev=max_norm_dev, energy_drift=float(np.max(np.abs(energies-energies[0]))), observables_approximation='factorized'))
