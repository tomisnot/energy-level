"""Read-only monitor store, finite JSON, static path boundary and lifecycle."""
import sys
import json
import tempfile
import pickle
from pathlib import Path
from http.client import HTTPConnection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mecha_v2.run_monitor import RunMonitorStore, RunMonitorServer

store = RunMonitorStore(max_records=1)
first = store.begin(dict(method='ed'))
second = store.begin(dict(method='tdvp'))
assert len(store.snapshot()['runs']) == 2  # active runs are never evicted
store.attach_progress(second, [.25, 0])
assert store.snapshot()['runs'][0]['progress'] == 25
store.finish(first, summary=dict(ok=True, gap=float('nan')))
third = store.begin(dict(method='ed'))
records = store.snapshot()['runs']
assert first not in [r['id'] for r in records]
assert {second,third} == {r['id'] for r in records}

with tempfile.TemporaryDirectory(prefix='el-monitor-') as folder:
    base = Path(folder).resolve()
    assert base.parent == Path(tempfile.gettempdir()).resolve()
    root = base/'web'; root.mkdir()
    (root/'index.html').write_text('<p>monitor</p>',encoding='utf-8')
    (base/'outside.txt').write_text('outside',encoding='utf-8')
    port_file = base/'port'
    store=RunMonitorStore(max_records=8)
    rid=store.begin(config={})
    store.attach_progress(rid,[0.,0,''])
    assert store.snapshot()['runs'][0]['progress']==0
    aid='a1b2c3d4e5f6'
    full=pickle.dumps(dict(native_result=[1,2,3]))
    (base/(aid+'.pkl')).write_bytes(full)
    store.update_run(rid,dict(method='ed'))
    store.finish(rid,summary=dict(ok=True,artifact_id=aid),raw=dict(t=[0.,1.],pops=[[1.,0.],[.5,.5]]))
    invalid=store.begin();store.finish(invalid,summary=dict(ok=True,artifact_id='../outside'))
    cancelled=store.begin();store.finish(cancelled,error=RuntimeError('cancelled'))
    assert store.snapshot()['runs'][0]['status']=='cancelled'
    cancelled=store.begin();store.attach_progress(cancelled,[.3,1,1])
    store.finish(cancelled,summary=dict(ok=False))
    assert store.snapshot()['runs'][0]['status']=='cancelled'
    server = RunMonitorServer(store,root,lambda:dict(gap=float('nan')),port_file=port_file,artifact_root=base).start()
    try:
        assert server.start() is server
        for path,status in [('/',200),('/api/runs',200),('/api/config',200),
                            ('/../outside.txt',404),('/%2e%2e/outside.txt',404),
                            ('/..%5coutside.txt',404),
                            (f'/api/runs/{rid}/result',200),
                            ('/api/runs/missing/result',404),
                            (f'/api/runs/{invalid}/artifact',404)]:
            connection=HTTPConnection('127.0.0.1',server.port,timeout=5)
            connection.request('GET',path)
            response=connection.getresponse(); payload=response.read()
            assert response.status==status,(path,response.status)
            if path.startswith('/api/'):
                assert b'NaN' not in payload
                json.loads(payload)
                if path=='/api/runs':
                    assert b'datasets' not in payload
                if path==f'/api/runs/{rid}/result':
                    assert json.loads(payload)['visualization']['artifact_available']
            connection.close()
        connection=HTTPConnection('127.0.0.1',server.port,timeout=5)
        connection.request('GET',f'/api/runs/{rid}/artifact')
        response=connection.getresponse()
        assert response.status==200 and response.read()==full
        connection.close()
        connection=HTTPConnection('127.0.0.1',server.port,timeout=5)
        connection.request('POST','/api/runs',body='{}')
        response=connection.getresponse();response.read()
        assert response.status==501
        connection.close()
        port_file.write_text('another server',encoding='utf-8')
    finally:
        server.stop(); server.stop()
    assert port_file.read_text(encoding='utf-8')=='another server'
print('[run_monitor] store / progress / finite JSON / lazy results / artifacts / read-only HTTP: PASS')
