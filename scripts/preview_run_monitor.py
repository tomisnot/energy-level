"""Small real AI-path computations served in the run observer for visual QA.

No numerical demo curves: every card is produced by ElEngine.run and core.
Ctrl+C closes this isolated preview server. Production records are untouched.
"""
import argparse
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mecha_v2.engine import ElEngine
from mecha_v2.bridge import ElOps
from mecha_v2.run_monitor import RunMonitorServer
from mecha.surface import ExecutionContext


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=0)
    args=parser.parse_args()
    engine=ElEngine(ROOT)
    folder=ROOT/'.mecha-v2'/'visual-preview'
    engine.artifact_writer=lambda run,res:ElOps._land(SimpleNamespace(root=folder),run,res)
    atoms=[dict(g=[60,0,.5,.5],r=[60,1,1.5,1.5],pos=[6.*(i%3),6.*(i//3),0.],
                Omega_mhz=1.1+.1*i,Omega_phase_rad=i*.5,delta_mhz=.2*(i-2))
           for i in range(6)]
    config={'dynamic.mode':2,'dynamic.t_max':1200.,'dynamic.npts':121,
        'manybody.atoms':atoms,'manybody.solver':'ed','manybody.n_window':0,
        'manybody.second_order':False,'manybody.B_gauss':21.7,
        'manybody.loops':[[0,1,4,3],[1,2,5,4]],'manybody.return_state':True}
    server=RunMonitorServer(engine.run_monitor,ROOT/'web'/'run_monitor',lambda:config,
                            port=args.port,artifact_root=folder).start()
    print(f'REAL_RUN_PREVIEW=http://127.0.0.1:{server.port}/',flush=True)
    stop=threading.Event()
    signal.signal(signal.SIGINT,lambda *_:stop.set())
    try:
        for want,reason in [('gs','六原子 · 基态分布'),('thermal','六原子 · 有限温扫描'),
                            ('dos','六原子 · 态密度'),('', '六原子 · 相干演化与关联')]:
            request=dict(config,**{'manybody.want':want,'manybody.eq_backend':'ed',
                'manybody.betas':[.1,.2,.5,1.,2.],'manybody.return_state':want in ('','gs')})
            result=engine.run(request,context=ExecutionContext(),reason=reason)
            print(reason,result.get('ok'),result.get('monitor_id'),flush=True)
        print('All preview cards are actual small-system results.',flush=True)
        stop.wait()
    finally:
        server.stop()
        if engine._pool is not None:
            engine._pool.shutdown(wait=True)


if __name__=='__main__':
    main()
