"""只读运行监控：把真实 RunSpec/进度/摘要投影给本地 web 卡片。"""
from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote
from .visualization import project_result, project_drives, model_view


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def _psi0_label(run):
    """初态的人话描述（优先用 RunSpec 的 quench 元数据，而不是落地后的 kind）。"""
    run = run or {}
    prepared = run.get('quench') or {}
    psi0 = run.get('psi0')
    kind = prepared.get('kind')
    landed = 'MPS' if (isinstance(psi0, dict) and psi0.get('kind') == 'mps') else '态矢'
    if kind == 'permutation_sector':
        weight = prepared.get('projection_weight')
        return '置换扇区 sign=%+d（投影权重 %s）→ %s' % (
            int(prepared.get('sign') or 0),
            '—' if weight is None else ('%.4g' % float(weight)), landed)
    if kind == 'ground_of':
        e0 = prepared.get('E0')
        return '另一模型基态%s → %s' % (
            '' if e0 is None else '（E₀=%.6g）' % float(e0), landed)
    if isinstance(psi0, dict):
        if psi0.get('kind') == 'mps':
            return '原生 MPS（%s）' % (psi0.get('backend') or '—')
        if psi0.get('kind'):
            return str(psi0['kind'])
    if psi0 is None:
        return '|g…g⟩'
    if isinstance(psi0, str):
        return psi0
    if hasattr(psi0, 'shape'):
        return '态矢（shape=%s）' % (list(psi0.shape),)
    return str(psi0)




class RunMonitorStore:
    """运行卡片：内存视图 **+ 磁盘持久化**。

    磁盘是真相：每次运行落一对文件 ``<id>.meta.json``（小，列表用）与
    ``<id>.viz.json``（大，选中时按需读）。进程启动时只读**最新 max_records 张
    的元数据**，所以"什么时候启动都显示磁盘上最新的 N 条"。

    性能约束（卡片页每 ~2 s 轮询一次 ``/api/runs``）：

    - ``snapshot()`` **不碰磁盘**（启动时加载一次，之后内存权威）；
    - 可视化载荷只在 ``result(id)`` 时读，并按 ``max_view_bytes`` 做 LRU 缓存；
    - 单张卡片载荷超过 ``max_card_bytes`` 就不落 viz（只留元数据），避免大数组
      把磁盘/内存撑爆；
    - 磁盘总量超过 ``max_disk_bytes``（默认 1 GiB）时，从最老的卡片开始删。
    """
    def __init__(self, max_records=30, *, card_dir=None,
                 max_disk_bytes=1024 ** 3, max_view_bytes=128 * 1024 ** 2,
                 max_card_bytes=8 * 1024 ** 2):
        self._lock = threading.RLock()
        self._records = {}
        self._max_records = int(max_records)
        if self._max_records < 1:
            raise ValueError('max_records must be positive')
        self._card_dir = Path(card_dir) if card_dir is not None else None
        self._max_disk_bytes = int(max_disk_bytes)
        self._max_view_bytes = int(max_view_bytes)
        self._max_card_bytes = int(max_card_bytes)
        self._views = {}                 # rid -> (bytes, payload, tick)
        self._view_bytes = 0
        self._tick = 0
        if self._card_dir is not None:
            self.configure(self._card_dir)

    # ---- 磁盘层：元数据 / 载荷 / 上限 ---------------------------------

    def configure(self, card_dir, *, max_disk_bytes=None, max_records=None):
        """接上卡片目录并加载最新 N 张（幂等；启动时调一次）。"""
        with self._lock:
            self._card_dir = Path(card_dir)
            if max_disk_bytes is not None:
                self._max_disk_bytes = int(max_disk_bytes)
            if max_records is not None:
                self._max_records = int(max_records)
            self._card_dir.mkdir(parents=True, exist_ok=True)
            self._load()
            self._prune_disk()                     # 启动时按磁盘预算清一次
        return self

    def _meta_path(self, rid):
        return self._card_dir / f'{rid}.meta.json' if self._card_dir else None

    def _viz_path(self, rid):
        return self._card_dir / f'{rid}.viz.json' if self._card_dir else None

    @staticmethod
    def _meta_of(rec):
        out = {k: v for k, v in rec.items() if not k.startswith('_')}
        out['has_visualization'] = bool(rec.get('_has_viz'))
        return out

    def _write_meta(self, rec):
        path = self._meta_path(rec['id'])
        if path is None:
            return
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(_jsonable(self._meta_of(rec)), ensure_ascii=False,
                                  allow_nan=False), encoding='utf-8')
        tmp.replace(path)

    def _write_view(self, rid, payload):
        path = self._viz_path(rid)
        if path is None:
            return 0
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        size = len(data.encode('utf-8'))
        if size > self._max_card_bytes:
            return 0                     # 单张太大：不落盘，只留元数据
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(data, encoding='utf-8')
        tmp.replace(path)
        return size

    def _load(self):
        """读磁盘上最新 max_records 张的**元数据**（不读 viz 载荷）。"""
        if self._card_dir is None or not self._card_dir.is_dir():
            return
        rows = []
        for path in self._card_dir.glob('*.meta.json'):
            try:
                meta = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            rid = str(meta.get('id') or path.name[:-len('.meta.json')])
            if not rid or rid in self._records:
                continue
            rows.append((float(meta.get('created_epoch') or path.stat().st_mtime), rid, meta))
        rows.sort(key=lambda r: r[0])              # 旧 → 新（snapshot 会再倒序）
        for epoch, rid, meta in rows[-self._max_records:]:
            rec = dict(meta)
            rec.update({'id': rid, 'created_epoch': epoch,
                        'status': str(meta.get('status') or 'completed'),
                        '_visualization': None,
                        '_has_viz': bool(meta.get('has_visualization')),
                        '_progress_arr': None, '_started': time.monotonic()})
            self._records[rid] = rec

    def _prune_disk(self):
        """磁盘上限：按最老优先删（meta 与 viz 成对）。"""
        if self._card_dir is None:
            return
        files = [p for p in self._card_dir.glob('*.json') if p.is_file()]
        groups = {}
        for p in files:
            rid = p.name.rsplit('.', 3)[0]
            groups.setdefault(rid, []).append(p)
        total = sum(p.stat().st_size for p in files)
        if total <= self._max_disk_bytes:
            return
        order = sorted(groups.items(),
                       key=lambda kv: max(p.stat().st_mtime_ns for p in kv[1]))
        for rid, paths in order:
            if total <= self._max_disk_bytes:
                break
            for p in paths:
                try:
                    total -= p.stat().st_size
                    p.unlink()
                except OSError:
                    pass
            self._views.pop(rid, None)

    def _view_for(self, rid):
        """按需读一张卡的载荷，带 LRU 字节预算。"""
        hit = self._views.get(rid)
        if hit is not None:
            self._tick += 1
            self._views[rid] = (hit[0], hit[1], self._tick)
            return hit[1]
        path = self._viz_path(rid)
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        size = path.stat().st_size
        if size <= self._max_view_bytes:
            self._tick += 1
            self._views[rid] = (size, payload, self._tick)
            self._view_bytes += size
            while self._view_bytes > self._max_view_bytes and len(self._views) > 1:
                oldest = min(self._views, key=lambda k: self._views[k][2])
                self._view_bytes -= self._views[oldest][0]
                self._views.pop(oldest, None)
        return payload


    @staticmethod
    def _view(run, config):
        config = config or {}
        keys = ('version','nbody','two_body','method','method_in','t_max','npts',
                'n_atoms','dim','full_dim','equilibrium','want','betas','states',
                'labels','n_waves','pair','engine_kwargs','warnings','psi0_A',
                'psi0_B','psi0_pair','terms','plan_summary','decays','decays_A','decays_B',
                'quench', 'template', 'calib_summary', 'build', 'build_receipt', 'label', 'notes')
        view = {key:_jsonable(run[key]) for key in keys if key in run}
        # 几何取本次 RunSpec 的 calib（多体唯一模型）；单/两体没有 calib 时留空。
        view['atoms'] = _jsonable((run.get('calib') or {}).get('atoms') or [])
        # Only the small, JSON-facing configuration is exposed. Native state
        # tensors and vectors belong in artifacts, not a polling response.
        view['config'] = {key:_jsonable(value) for key,value in config.items()
                          if isinstance(value,(str,int,float,bool,type(None)))}
        if isinstance(run.get('psi0'),dict):
            view['psi0'] = {key:_jsonable(run['psi0'].get(key))
                            for key in ('kind','backend','order','charge_sector')}
        elif hasattr(run.get('psi0'),'shape'):
            view['psi0'] = {'kind':'vector','shape':list(run['psi0'].shape)}
        else:
            view['psi0'] = _jsonable(run.get('psi0'))
        # 磁场只属于单/两体的物理设置；多体模型里没有"场"这个概念（场已被标定进 calib）。
        view['field'] = (None if run.get('nbody') == 'many' else
                         {'B_gauss': config.get('field.B_gauss', 0),
                          'E_vpcm': config.get('field.E_vpcm', 0)})
        view['psi0_label'] = _psi0_label(run)
        try:
            view['drives_preview']=project_drives(run,config)
        except (ValueError,TypeError,KeyError,AssertionError) as exc:
            view['drives_preview']=[]
            view['drive_preview_note']=str(exc)
        return view

    def begin(self, run=None, config=None, *, actor="ai", reason=""):
        rid = "run-" + uuid.uuid4().hex[:12]
        rec = {"id":rid,"status":"running","createdAt":time.strftime("%H:%M:%S"),
               "progress":0,"stage":"准备","elapsed":"—","actor":actor,"reason":reason,
               "created_epoch":time.time(),"run":self._view(run or {},config),"result":None,"error":None,
               "has_visualization":False,"result_version":0,"input_version":0,
               "_progress_arr":None,"_started":time.monotonic(),"_visualization":None}
        with self._lock:
            self._records[rid]=rec
            self._trim()
        self._write_meta(rec)                       # 启动/运行中都落盘：重启后仍可见
        return rid

    def _trim(self):
        while len(self._records)>self._max_records:
            finished=next((key for key,item in self._records.items() if item['status']!='running'),None)
            if finished is None:
                break
            self._records.pop(finished)

    def update_run(self,rid,run,config=None):
        with self._lock:
            rec = self._records.get(rid)
            if rec is None:
                return
            rec['run']=self._view(run,config)
            # 模型本体只留在**内存**里：结果投影要画哈密顿量（耦合矩阵/δ），
            # 但它不该进轮询快照，也不落盘（重启后旧卡没有它，前端如实降级）。
            rec['_calib']=run.get('calib') if isinstance(run,dict) else None
            rec['input_version']+=1
            snapshot = dict(rec)
        self._write_meta(snapshot)                  # 文件 I/O 不占锁

    def is_running(self,rid):
        with self._lock:
            return self._records.get(rid,{}).get('status')=='running'

    def attach_progress(self, rid, progress_arr):
        with self._lock:
            if rid in self._records:
                self._records[rid]["_progress_arr"] = progress_arr

    def finish(self,rid,summary=None,raw=None,error=None):
        # Build the numerical projection outside the store lock. Rendering
        # errors cannot turn successful physics into a failed computation.
        visualization = None
        projection_error = None
        if raw is not None:
            with self._lock:
                rec = self._records.get(rid, {})
                view = dict(rec.get('run', {}))
                calib = rec.get('_calib')
            if calib:
                view['calib'] = calib
            try:
                visualization=_jsonable(project_result(view,raw))
                visualization['artifact_id']=(summary or {}).get('artifact_id')
            except Exception as exc:
                projection_error=str(exc)
        with self._lock:
            rec=self._records.get(rid)
            if rec is None:
                return
            cancelled = (error is not None and ('取消' in str(error) or 'cancel' in str(error).lower()))
            failed = error is not None or (summary is not None and summary.get('ok') is False)
            arr=rec.get('_progress_arr')
            cancelled = cancelled or bool(failed and arr is not None and len(arr)>1 and arr[1])
            rec.update(status='cancelled' if cancelled else 'failed' if failed else 'completed',
                progress=rec.get('progress',0) if failed else 100,
                stage='已取消' if cancelled else '失败' if failed else '已完成',
                elapsed=f"{time.monotonic()-rec['_started']:.1f}s",
                result=_jsonable(summary or {}),
                error=str(error) if error else (summary or {}).get('hint') if failed else None,
                has_visualization=visualization is not None,projection_error=projection_error,
                result_version=rec['result_version']+1,_visualization=visualization,
                _has_viz=visualization is not None,_progress_arr=None)
            self._trim()
            snapshot = dict(rec)
        # 落盘：先写（可能很大的）载荷，再写元数据；最后按 1 GiB 上限回收。
        if visualization is not None:
            self._write_view(rid, visualization)
        self._write_meta(snapshot)
        self._prune_disk()

    def result(self,rid):
        with self._lock:
            rec=self._records.get(rid)
            viz=rec.get('_visualization') if rec is not None else None
            load_disk=bool(rec is not None and viz is None and rec.get('_has_viz'))
            summary=rec.get('result') if rec is not None else None
        if load_disk:
            viz=self._view_for(rid)        # 磁盘按需读（LRU 字节预算）
        if rec is None and viz is None:
            return None
        return dict(ok=True,id=rid,visualization=viz,summary=summary or {})

    def model(self, rid):
        """按需取**模型本体**（耦合矩阵/δ 的可视化数据）。只在内存里，不落盘、不进轮询。"""
        with self._lock:
            rec = self._records.get(rid)
            calib = rec.get('_calib') if rec is not None else None
        if not calib:
            return None
        return {'ok': True, 'id': rid, 'model': model_view(calib)}

    def snapshot(self, config=None):
        with self._lock:
            out = []
            for rec in self._records.values():
                item = {k: v for k, v in rec.items() if not k.startswith("_")}
                arr = rec.get("_progress_arr")
                if rec["status"] == "running" and arr is not None:
                    try:
                        frac = float(arr[0])
                        try:
                            stage = int(float(arr[2])) if len(arr) > 2 else 1
                        except (ValueError, TypeError):
                            stage = 0 if not rec['input_version'] else 1
                        percent=frac*100
                        if len(arr)>2:
                            percent=(frac*12 if stage==0 else 12+86*frac if stage==1 else 98+2*frac)
                        item["progress"] = round(max(0, min(100, percent)), 1)
                        item["stage"] = ("准备" if stage == 0 else "积分" if stage == 1 else "收尾")
                        item["elapsed"] = f"{time.monotonic() - rec['_started']:.1f}s"
                    except Exception:
                        pass
                out.append(item)
            return {"ok": True, "runs": list(reversed(out)), "config": {}}


class RunMonitorServer:
    def __init__(self, store, root, config_reader, host="127.0.0.1", port=0, port_file=None,
                 artifact_root=None):
        self.store, self.root, self.config_reader = store, Path(root), config_reader
        self.host, self.port, self.port_file = host, int(port), Path(port_file) if port_file else None
        self.httpd = None
        self._thread = None
        self.artifact_root=Path(artifact_root).resolve() if artifact_root is not None else None

    def _artifact(self,rid):
        result=self.store.result(rid)
        aid=((result or {}).get('summary') or {}).get('artifact_id')
        if not self.artifact_root or not isinstance(aid,str) or not re.fullmatch('[a-f0-9]{12}',aid):
            return None
        path=(self.artifact_root/(aid+'.pkl')).resolve()
        return path if path.is_relative_to(self.artifact_root) and path.is_file() else None

    def start(self):
        if self.httpd is not None:
            return self
        store, root, config_reader = self.store, self.root.resolve(), self.config_reader
        endpoint=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return
            def _send(self, code, payload, content_type="application/json; charset=utf-8"):
                data = payload if isinstance(payload, bytes) else (json.dumps(_jsonable(payload), ensure_ascii=False, allow_nan=False).encode() if not isinstance(payload, str) else payload.encode())
                self.send_response(code); self.send_header("Content-Type", content_type); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            def do_GET(self):
                path = unquote(urlparse(self.path).path)
                if path == "/api/runs": self._send(200, store.snapshot()); return
                if path.startswith('/api/runs/') and path.endswith('/model'):
                    rid=path[len('/api/runs/'):-len('/model')]
                    payload=store.model(rid)
                    self._send(200 if payload else 404,
                               payload or {'ok':False,'error':'not_found'}); return
                if path.startswith('/api/runs/') and path.endswith('/result'):
                    rid=path[len('/api/runs/'):-len('/result')]
                    result=store.result(rid)
                    if result and result.get('visualization'):
                        result=dict(result,visualization=dict(result['visualization'],
                                    artifact_available=endpoint._artifact(rid) is not None))
                    self._send(200 if result else 404,result or {'ok':False,'error':'not_found'}); return
                if path.startswith('/api/runs/') and path.endswith('/artifact'):
                    rid=path[len('/api/runs/'):-len('/artifact')]
                    artifact=endpoint._artifact(rid)
                    if artifact is None:
                        self._send(404,{'ok':False,'error':'not_found'}); return
                    self.send_response(200)
                    self.send_header('Content-Type','application/octet-stream')
                    self.send_header('Content-Disposition','attachment; filename="EL-'+rid+'.pkl"')
                    self.send_header('Content-Length',str(artifact.stat().st_size))
                    self.end_headers()
                    with artifact.open('rb') as stream:
                        while chunk:=stream.read(65536):
                            self.wfile.write(chunk)
                    return
                if path == "/api/config": self._send(200, {"ok": True, "config": _jsonable(config_reader())}); return
                target = (root / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
                if not target.is_relative_to(root):
                    self._send(404, {"ok": False, "error": "not_found"}); return
                if target.is_file(): self._send(200, target.read_bytes(), "text/html; charset=utf-8" if target.suffix == ".html" else "text/css; charset=utf-8" if target.suffix == ".css" else "application/javascript; charset=utf-8"); return
                self._send(404, {"ok": False, "error": "not_found"})
        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self.httpd.server_address[1]
        if self.port_file: self.port_file.write_text(str(self.port), encoding="utf-8")
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self.httpd:
            self.httpd.shutdown(); self.httpd.server_close()
            self.httpd = None
        if self._thread:
            self._thread.join(timeout=2.)
            self._thread = None
        if self.port_file and self.port_file.exists():
            try:
                if self.port_file.read_text(encoding='utf-8').strip() == str(self.port):
                    self.port_file.unlink()
            except OSError: pass
