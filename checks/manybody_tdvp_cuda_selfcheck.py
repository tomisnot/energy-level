"""N24 CUDA TDVP product contracts, continuation and independent CPU comparison."""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from threadpoolctl import threadpool_limits
from checks.checklib import Checker
from src.core import protocol
from src.core.manybody.backends import ConfigConflict
from src.core.manybody.tdvp_dense import evolve_tdvp_dense
from src.core.manybody.tdvp_cuda import evolve_tdvp_cuda

C=Checker('manybody_tdvp_cuda')
ROOT=Path(__file__).resolve().parents[1]


def parameters():
    cal=json.loads((ROOT/'checks/fixtures/kagome_n24.json').read_text(encoding='utf-8'))
    return {'manybody.calib':cal,'dynamic.mode':2,'dynamic.t_max':2.,'dynamic.npts':3,
            'manybody.solver':'tdvp-cuda','manybody.psi0':'rg'*12,
            'manybody.chi_max':16,'manybody.max_step_ns':1.,'manybody.dtype':'float32',
            'manybody.svd_device':'cpu','manybody.return_state':True,
            'manybody.want_observables':True,'manybody.mps_order':list(reversed(range(24))),
            'manybody.pauli_strings':[{'label':'xyz','terms':[[1,'x'],[5,'y'],[9,'z']]}]}


def main():
    p=parameters()
    C.check('schema advertises tdvp-cuda','tdvp-cuda' in protocol.config_schema()['manybody.solver']['enum'])
    for extra in ({'manybody.charge_sector':12},{'manybody.constraint':{'kind':'blockade','mode':'project','edges':[[0,1]]}}):
        try:
            protocol.build_run(None,dict(p,**extra))
        except (ConfigConflict,ValueError):
            C.check('unsupported model rejected',True)
        else:
            C.check('unsupported model rejected',False)
    try:
        import torch
    except ImportError:
        C.skip('PyTorch unavailable')
        return 0 if C.summary() else 1
    if not torch.cuda.is_available():
        C.skip('CUDA hardware unavailable')
        return 0 if C.summary() else 1
    from scripts.tdvp_fast import torch_tdvp as legacy_kernel
    from src.core.manybody import tdvp_cuda_kernel
    C.check('prototype and product share the same kernel',legacy_kernel.sweep is tdvp_cuda_kernel.sweep)
    with threadpool_limits(limits=2):
        out=protocol.evolve_run(protocol.build_run(None,p))
        run=protocol.build_run(None,p)
        ref=evolve_tdvp_dense(run['calib'],psi0=run['psi0'],Om=run['Om'],delta=run['delta'],
                             t_grid=run['t'],chi_max=16,max_step_ns=1.,mps_order=p['manybody.mps_order'],
                             want_observables=True,pauli_strings=p['manybody.pauli_strings'])
        err=float(np.max(np.abs(out['pops_atom']-ref['pops_atom'])))
        perr=float(np.max(np.abs(out['observables']['pauli_strings']-ref['observables']['pauli_strings'])))
        C.group('N24 standard RunSpec, basis/order and observables',[
            ('backend',out['backend']=='tdvp-cuda'),('pops vs CPU, c64 budget 1e-5',err<1e-5),
            ('Pauli vs CPU',perr<2e-6),('shape',out['pops_atom'].shape==(3,24)),
            ('schema','pops_atom' in out['data_schema']['quantities']),
            ('SVD report',out['error_report']['svd_device']=='cpu'),
            ('complex64',out['error_report']['dtype']=='complex64')],detail=f'dp={err}, dPauli={perr}')
        from tenpy.tools import hdf5_io
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.h5'
            hdf5_io.save(out['state'],str(path))
            state=hdf5_io.load(str(path))
            direct=evolve_tdvp_cuda(run['calib'],psi0=state,t_grid=[2.,3.],Om=run['Om'],delta=run['delta'],
                                   chi_max=16,max_step_ns=1.,dtype='float32',mps_order=state['order'],
                                   want_observables=False)
        whole=protocol.evolve_run(protocol.build_run(None,dict(p,**{'dynamic.t_max':3.,'dynamic.npts':4})))
        C.check('N24 HDF5 continuation',np.max(np.abs(direct['pops_atom'][-1]-whole['pops_atom'][-1]))<2e-6)
        seg=[{'t0':0.,'Om':[0.]*24,'delta':[0.]*24},
             {'t0':1.,'Om':[.3]*24,'phase':[.4]*24,'delta':[0.]*24},
             {'t0':2.,'Om':[.6]*24,'phase':[.8]*24,'delta':[0.]*24}]
        ps=dict(p,**{'manybody.drive_segments':seg,'manybody.drive_interp':'midpoint',
                     'manybody.dtype':'float64','manybody.truncerr':1e-8})
        sr=protocol.build_run(None,ps)
        so=protocol.evolve_run(sr)
        rf=evolve_tdvp_dense(sr['calib'],psi0=sr['psi0'],t_grid=sr['t'],Om=sr['Om'],delta=sr['delta'],
                             segments=sr['segments'],drive_interp='midpoint',chi_max=16,max_step_ns=1.,truncerr=1e-8,
                             mps_order=p['manybody.mps_order'],want_observables=False)
        se=float(np.max(np.abs(so['pops_atom']-rf['pops_atom'])))
        C.group('N24 complex128, midpoint, phase/Omega-only drive changes',[
            ('two changed fields',so['error_report']['segment_model_builds']==2),
            ('dtype',so['error_report']['dtype']=='complex128'),('CPU cross-check, global budget 1e-6',se<1e-6)],detail=f'dp={se}')
        # Independent N24 onsite-only oracle, including the physical phase convention.
        from scipy.linalg import expm
        from src.core.evolve_util import CONV
        analytic_cal={'atoms':[{'id':i,'pos':[float(i),0.,0.]} for i in range(24)],'pairs':[]}
        drives=[{'t0':0.,'Om':[3.]*24,'phase':[.4]*24,'delta':[.7]*24},
                {'t0':10.,'Om':[6.]*24,'phase':[.8]*24,'delta':[1.1]*24}]
        analytic=evolve_tdvp_cuda(analytic_cal,psi0='g'*24,t_grid=[0.,10.,20.],
                                  segments=drives,dtype='float64',max_step_ns=10.,
                                  chi_max=4,taylor_tol=1e-12,want_observables=True)
        local=np.array([1.,0.],complex)
        for segment in drives:
            omega=segment['Om'][0]*np.exp(1j*segment['phase'][0])
            h=np.array([[0.,np.conj(omega)/2],[omega/2,segment['delta'][0]]])
            local=expm(-1j*CONV*10.*h)@local
        expected_p=abs(local[1])**2
        expected_y=np.real(local.conj()@np.array([[0.,-1j],[1j,0.]])@local)
        C.group('N24 analytic independent-site oracle',[
            ('population',np.max(np.abs(analytic['pops_atom'][-1]-expected_p))<1e-10),
            ('complex phase',np.max(np.abs(analytic['observables']['sy'][-1]-expected_y))<1e-10)])
        co=evolve_tdvp_cuda(run['calib'],psi0='rg'*12,t_grid=[0.,1.],chi_max=8,max_step_ns=1.,
                            dtype='float32',svd_device='cuda',svd_driver='gesvd',want_observables=False)
        C.check('N24 explicit GPU SVD',co['error_report']['svd_device']=='cuda' and np.isfinite(co['pops_atom']).all())
        try:
            evolve_tdvp_cuda(run['calib'],psi0='rg'*12,t_grid=[0.,1.],progress_arr=[0.,1.])
        except RuntimeError as exc:
            C.check('cancelled before evolution','取消' in str(exc) or 'cancel' in str(exc).lower())
        else:
            C.check('cancelled before evolution',False)
    return 0 if C.summary() else 1


if __name__=='__main__':
    sys.exit(main())
