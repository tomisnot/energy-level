# -*- coding: utf-8 -*-
"""Cross-check generic time-evolution backends against independent formulas/ED."""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.linalg import expm
from checks.checklib import Checker
from src.core.evolve_util import CONV
from src.core.manybody.classical_spin import evolve_classical_spin
from src.core.manybody.engine_peps import evolve_peps
from src.core.manybody.engine_trajectories import evolve_trajectories
C=Checker('manybody_timeevolution')
# 1) one-site classical Bloch dynamics: H=Omega*sigma_x/2.
cal1={'atoms':[{'pos':(0,0)}],'pairs':[],'terms':[]}; t=np.linspace(0,100,11)
r=evolve_classical_spin(cal1,psi0='g',t_grid=t,Om=np.array([1.0]),delta=np.zeros(1),terms=[])
exact=np.sin(CONV*t/2)**2
C.close('classical-spin 一站点解析布居',float(np.max(np.abs(r['n_atom'][:,0]-exact))),0,atol=2e-8)
C.close('classical-spin 自旋范数',r['error_report']['max_spin_norm_dev'],0,atol=2e-12)
# 2) 1x2 PEPS against dense two-site XY propagator.
cal2={'atoms':[{'pos':(0,0)},{'pos':(1,0)}],'pairs':[{'i':0,'j':1,'V':0,'J':2.0,'A':0}],'terms':['I2']}
t2=np.array([0.,10.]); rp=evolve_peps(cal2,psi0='gr',t_grid=t2,Om=np.zeros(2),delta=np.zeros(2),terms=['I2'],max_step_ns=.1,chi_max=8)
H=np.zeros((4,4),complex); H[1,2]=H[2,1]=2.; psi=np.array([0,1,0,0],complex); U=expm(-1j*CONV*10*H); p=abs(U@psi)**2
C.close('PEPS 1x2 XY 与 ED',float(np.max(np.abs(rp['pops'][-1]-np.array([p[2],p[1]])))),0,atol=2e-7)
C.check('PEPS 真张量网络',rp['backend']=='peps-real-time' and rp['error_report']['shape']==[1,2])
# 3) local amplitude damping trajectories against exp(-gamma t).
cal3={'atoms':[{'pos':(0,0)},{'pos':(1,0)}],'pairs':[],'terms':[]}; tg=np.array([0.,10.])
rt=evolve_trajectories(cal3,psi0='rr',t_grid=tg,Om=np.zeros(2),delta=np.zeros(2),collapse_ops=[{'site':0,'operator':'M','rate':10}],n_trajectories=32,max_step_ns=1.,seed=7)
C.close('trajectories 局部衰减平均',float(rt['pops'][-1,0]),np.exp(-0.1),atol=0.12)
C.check('trajectories 给出 Monte Carlo SEM',rt['stderr'].shape==rt['pops'].shape and rt['error_report']['first_order_jump_split'])
sys.exit(0 if C.summary() else 1)
