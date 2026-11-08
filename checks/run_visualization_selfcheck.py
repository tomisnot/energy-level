"""Real small-system AI runs and bounded, identity-preserving Web projections."""
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mecha.surface import ExecutionContext
from mecha_v2.engine import ElEngine
from mecha_v2.bridge import ElOps
from mecha_v2.run_monitor import RunMonitorStore
from mecha_v2.visualization import (project_result, project_input_view,
                                    sampled_indices, MAX_LINKS)
from src.core import protocol
from src.core.manybody import runspec_mb
from src.core.manybody.results import describe_result


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.cal=dict(atoms=[dict(id=i,pos=[6.*(i%2),6.*(i//2),0.]) for i in range(3)],
                      pairs=[])
        self.spec={'dynamic.mode':2,'dynamic.t_max':3.,'dynamic.npts':17,
            'manybody.solver':'ed','manybody.return_state':True,
            'manybody.calib':{'atoms':[dict(id=a['id'],pos=a['pos'],Omega_mhz=1.+i,
                                           delta_mhz=.1*i)
                                      for i,a in enumerate(self.cal['atoms'])],
                              'pairs':[]},
            'manybody.loops':[[0,1,2]]}

    def test_real_ai_to_card_and_artifact(self):
        engine=ElEngine.__new__(ElEngine)
        engine._protocol=protocol; engine._cache=None; engine._pool=None
        engine.run_monitor=RunMonitorStore()
        with tempfile.TemporaryDirectory() as folder:
            engine.artifact_writer=lambda run,res:ElOps._land(SimpleNamespace(root=Path(folder)),run,res)
            for want in ('','gs','thermal','dos'):
                spec=dict(self.spec,**{'manybody.want':want,'manybody.eq_backend':'ed',
                    'manybody.betas':[.1,.3,1.], 'manybody.return_state':want in ('','gs')})
                out=engine.run(spec,context=ExecutionContext())
                self.assertTrue(out['ok'],out)
                detail=engine.run_monitor.result(out['monitor_id'])
                view=detail['visualization']
                self.assertEqual(len(view['sites']),3)
                self.assertEqual(view['artifact_id'],out['artifact_id'])
                with (Path(folder)/(out['artifact_id']+'.pkl')).open('rb') as stream:
                    _,raw=pickle.load(stream)
                self.assertIn('data_schema',raw)
                ds={d['id']:d for d in view['datasets']}
                if not want:
                    self.assertEqual(ds['pops_atom']['x'],raw['t'].tolist())
                    np.testing.assert_allclose(ds['pops_atom']['values'],raw['pops_atom'])
                    self.assertIn('entropy_partition',ds)
                    self.assertEqual(ds['entropy_partition']['part_a'],[0])
                    self.assertIn('local_bloch',ds)
                    self.assertEqual(ds['observables.link_corr']['kind'],'links')
                elif want=='thermal':
                    self.assertEqual(ds['E']['x'],[.1,.3,1.])
                elif want=='dos':
                    self.assertEqual(ds['rho']['kind'],'curve')
                    self.assertEqual(ds['rho']['x'][0],float(raw['energies'][0]))
                json.dumps(view,allow_nan=False)
                model=engine.run_monitor.model(out['monitor_id'])['model']
                self.assertEqual(model['initial_sz'],[1.0,1.0,1.0])
                self.assertEqual(model['initial_label'],'|g…g⟩')
            poll=engine.run_monitor.snapshot()['runs']
            self.assertEqual(len(poll),4)
            self.assertTrue(all(r['has_visualization'] for r in poll))
            self.assertNotIn('datasets',json.dumps(poll))
            # A build failure creates exactly one additional, failed card.
            with self.assertRaises(Exception):
                engine.run(dict(self.spec,**{'manybody.calib':{'atoms':[], 'pairs':[]}}),
                           context=ExecutionContext())
            self.assertEqual(len(engine.run_monitor.snapshot()['runs']),5)
            self.assertEqual(engine.run_monitor.snapshot()['runs'][0]['status'],'failed')

    def test_initial_sigma_z_any_state(self):
        from src.core.manybody.state import initial_sigma_z
        # 直积态
        np.testing.assert_allclose(initial_sigma_z('|grg>',3),[1,-1,1])
        # 叠加/纠缠态也能给出逐点均值，而不是猜一个直积态
        vec=np.zeros(8,complex);vec[0]=1/np.sqrt(2);vec[7]=1/np.sqrt(2)
        np.testing.assert_allclose(initial_sigma_z(vec,3),[0,0,0],atol=1e-12)
        # ground 需要求解器，展示层必须明确“不可用”
        self.assertIsNone(initial_sigma_z('ground',3))

    def test_complex_missing_and_exact_count_histogram(self):
        raw=describe_result(dict(t=np.array([0.,1.,9.]),pops_atom=np.ones((3,3))*.5,
            observables=dict(phase=np.array([[np.nan,0.,1.]]*3),
                link_corr=np.array([[1+2j],[2-3j],[0j]]),link_pairs=[[0,2]]),
            jump_counts=np.array([0,0,2,2,2,7])),self.cal)
        view=project_result({},raw)
        ds={d['id']:d for d in view['datasets']}
        self.assertIsNone(ds['observables.phase']['values'][0][0])
        self.assertEqual(ds['observables.link_corr']['imaginary'],[[2.],[-3.],[0.]])
        self.assertEqual(ds['jump_counts']['x'],[0.,2.,7.])
        self.assertEqual(ds['jump_counts']['values'],[2.,3.,1.])
        json.dumps(view,allow_nan=False)

    def test_basis_identity_and_sampling(self):
        # Sector indices are full-space identities, not vector column numbers.
        raw=dict(vec=np.array([.6,.8j]),basis=dict(indices=np.array([3,6])),
                 sites=[dict(site_id=i,pos=[i,0,0]) for i in range(3)])
        ds=project_result({},raw)['datasets'][0]
        self.assertEqual(ds['codes'],['6','3'])
        raw['basis']['indices']=np.array([2**60+1,2**60+2])
        self.assertEqual(project_result({},raw)['datasets'][0]['codes'][0],str(2**60+2))
        ids=sampled_indices(10001)
        self.assertEqual((int(ids[0]),int(ids[-1])),(0,10000))
        self.assertLessEqual(len(ids),160)
        n=50
        matrix=np.ones((n,n));matrix[2,3]=matrix[3,2]=10.
        view=project_result({},dict(corr=matrix,sites=[dict(pos=[i,0,0]) for i in range(n)]))
        links=next(d for d in view['datasets'] if d['id']=='corr')
        self.assertEqual(len(links['edges']),MAX_LINKS)
        self.assertIn([2,3],links['edges'])
        self.assertIn('最强',links['sampling'])

    def test_beta_site_projection_and_uncertainty(self):
        raw=describe_result(dict(betas=np.array([.2,1.]),n_atom=np.array([[.1,.2,.3],[.4,.5,.6]]),
            E=np.array([-3.,-4.]),E_err=np.array([.1,.2]),average_sign=np.array([1.,.9]),
            samples=np.array([10,20])),self.cal)
        ds={d['id']:d for d in project_result({},raw)['datasets']}
        self.assertEqual(ds['n_atom']['kind'],'sites')
        self.assertEqual(ds['n_atom']['axis'],'beta')
        self.assertEqual(ds['E']['stderr'],[.1,.2])
        self.assertEqual(ds['average_sign']['x'],[.2,1.])

    def test_browser_budget_is_visible_and_does_not_replace_missing_data(self):
        raw=describe_result(dict(t=np.array([0.,1.,9.]),
            pops_atom=np.ones((3,3))*.5,S=np.array([0.,np.nan,.4])),self.cal)
        with patch('mecha_v2.visualization.MAX_CELLS',6):
            view=project_result({},raw)
        ds={d['id']:d for d in view['datasets']}
        self.assertEqual(ds['pops_atom']['x'],[0.,9.])
        self.assertNotIn('S',ds)
        self.assertTrue(any('预算' in note for note in view['notes']))

    def test_pauli_strings_projection(self):
        vals = np.array([[.5 + .5j, -.2j], [.6 + .1j, -.1j], [.7 + 0j, .3j]])
        raw = describe_result(dict(
            t=np.array([0., 1., 2.]),
            observables=dict(pauli_strings=vals,
                             pauli_string_labels=['z@0', 'x@0 x@1'])), self.cal)
        ds = {d['id']: d for d in project_result({}, raw)['datasets']}
        first = ds['observables.pauli_strings.0']
        self.assertEqual(first['kind'], 'curve')
        self.assertEqual(first['x'], [0., 1., 2.])
        self.assertEqual(first['label'], 'Pauli 串期望 · z@0')
        self.assertAlmostEqual(first['values'][0], .5)
        self.assertAlmostEqual(first['imaginary'][0], .5)
        self.assertEqual(ds['observables.pauli_strings.1']['label'],
                         'Pauli 串期望 · x@0 x@1')
        json.dumps(project_result({}, raw), allow_nan=False)

    def test_generic_itemized_projection(self):
        vals = np.array([[.5, .2], [.6, .3], [.7, .4]], complex)
        raw = describe_result(dict(
            t=np.array([0., 1., 2.]),
            observables=dict(custom_strings=vals,
                             custom_string_labels=['a', 'b'])), self.cal)
        raw['data_schema']['quantities']['observables.custom_strings'] = dict(
            quantity='custom', axes=['time', 'item'], unit='1',
            items=dict(labels='observables.custom_string_labels'))
        ds = {d['id']: d for d in project_result({}, raw)['datasets']}
        self.assertTrue(ds['observables.custom_strings.0']['label'].endswith(' · a'))
        self.assertEqual(ds['observables.custom_strings.1']['x'], [0., 1., 2.])

    def test_input_view_segments(self):
        run = dict(nbody='many', n_atoms=2, t_max=1.,
                   Om=np.array([0., 0.], complex), delta=np.array([0., 0.]),
                   segments=[dict(t0=0., Om=[0.5, 0.5], delta=[0., 0.]),
                             dict(t0=.5, Om=[0., 0.], delta=[.2, .2])],
                   engine_kwargs=dict(pauli_strings=[[[0, 'x']], [[1, 'z']]]))
        view = project_input_view(run, {})
        self.assertEqual(view['drive']['kind'], 'segments')
        self.assertEqual(view['drive']['n_segments'], 2)
        self.assertEqual([s['t0'] for s in view['drive']['segments']], [0., .5])
        self.assertEqual(view['observables']['pauli_count'], 2)
        self.assertEqual(len(view['drives_preview']), 4)
        card = RunMonitorStore._view(run, {})
        self.assertEqual(card['drive']['n_segments'], 2)
        self.assertEqual(card['observables']['pauli_count'], 2)

    def test_projection_failure_does_not_fail_physics(self):
        store=RunMonitorStore()
        rid=store.begin()
        with patch('mecha_v2.run_monitor.project_result',side_effect=ValueError('bad plot metadata')):
            store.finish(rid,summary=dict(ok=True),raw=dict(t=[0.,1.]))
        record=store.snapshot()['runs'][0]
        self.assertEqual(record['status'],'completed')
        self.assertFalse(record['has_visualization'])
        self.assertIn('bad plot metadata',record['projection_error'])


if __name__=='__main__':
    unittest.main()
