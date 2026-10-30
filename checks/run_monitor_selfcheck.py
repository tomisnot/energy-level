"""Read-only monitor store, finite JSON, static path boundary and lifecycle."""
import sys
import json
import tempfile
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
    server = RunMonitorServer(store,root,lambda:dict(gap=float('nan')),port_file=port_file).start()
    try:
        assert server.start() is server
        for path,status in [('/',200),('/api/runs',200),('/api/config',200),
                            ('/../outside.txt',404),('/%2e%2e/outside.txt',404),
                            ('/..%5coutside.txt',404)]:
            connection=HTTPConnection('127.0.0.1',server.port,timeout=5)
            connection.request('GET',path)
            response=connection.getresponse(); payload=response.read()
            assert response.status==status,(path,response.status)
            if path.startswith('/api/'):
                assert b'NaN' not in payload
                json.loads(payload)
            connection.close()
        port_file.write_text('another server',encoding='utf-8')
    finally:
        server.stop(); server.stop()
    assert port_file.read_text(encoding='utf-8')=='another server'
print('[run_monitor] PASS=6 FAIL=0 SKIP=0')
