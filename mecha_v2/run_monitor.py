"""只读运行监控：把真实 RunSpec/进度/摘要投影给本地 web 卡片。"""
from __future__ import annotations

import json
import math
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote


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


def _plot_payload(run, res, limit=160):
    """只保留主要态的降采样曲线，避免把完整 pops 推到浏览器。"""
    try:
        import numpy as np
        arr = np.asarray((res or {}).get("pops"))
        if arr.ndim != 2 or arr.size == 0:
            return {}
        labels = [str(x.get("label", "?")) if isinstance(x, dict) else str(x)
                  for x in (res or {}).get("states", run.get("states") or [])]
        avg = arr.mean(axis=0)
        order = np.argsort(-avg)[:6]
        stride = max(1, int(np.ceil(arr.shape[0] / limit)))
        return {"series": [{"label": labels[int(i)] if int(i) < len(labels) else f"state-{int(i)}",
                             "values": [round(float(x), 7) for x in arr[::stride, int(i)]]}
                            for i in order],
                "n_t": int(arr.shape[0])}
    except Exception:
        return {}


class RunMonitorStore:
    def __init__(self, max_records=32):
        self._lock = threading.RLock()
        self._records = {}
        self._max_records = int(max_records)
        if self._max_records < 1:
            raise ValueError('max_records must be positive')

    def begin(self, run, config=None, *, actor="gui", reason=""):
        rid = "run-" + uuid.uuid4().hex[:12]
        states = run.get("states") or []
        view = {
            "version": run.get("version", 1), "nbody": run.get("nbody"),
            "two_body": bool(run.get("two_body")), "method": run.get("method"),
            "method_in": run.get("method_in"), "t_max": run.get("t_max"),
            "npts": run.get("npts"), "n_atoms": run.get("n_atoms"),
            "dim": run.get("dim"), "states": _jsonable(states),
            "labels": _jsonable(run.get("labels") or []),
            "n_waves": _jsonable(run.get("n_waves") or {}),
            "pair": _jsonable(run.get("pair")),
            "atoms": _jsonable((config or {}).get("manybody.atoms") or run.get("atoms") or []),
            "engine_kwargs": _jsonable(run.get("engine_kwargs") or {}),
            "warnings": _jsonable(run.get("warnings") or []),
            "config": _jsonable(config or {}),
        }
        rec = {"id": rid, "status": "running", "createdAt": time.strftime("%H:%M:%S"),
               "progress": 0, "stage": "准备", "elapsed": "—", "eta": "",
               "actor": actor, "reason": reason, "run": view, "result": None,
               "error": None, "_progress_arr": None, "_started": time.monotonic()}
        with self._lock:
            self._records[rid] = rec
            while len(self._records) > self._max_records:
                finished = next((key for key, item in self._records.items()
                                 if item['status'] != 'running'), None)
                if finished is None:
                    break
                self._records.pop(finished)
        return rid

    def attach_progress(self, rid, progress_arr):
        with self._lock:
            if rid in self._records:
                self._records[rid]["_progress_arr"] = progress_arr

    def finish(self, rid, summary=None, raw=None, error=None):
        with self._lock:
            rec = self._records.get(rid)
            if rec is None:
                return
            rec["status"] = "failed" if error else "completed"
            rec["progress"] = 100 if not error else rec.get("progress", 0)
            rec["stage"] = "失败" if error else "已完成"
            rec["elapsed"] = f"{time.monotonic() - rec['_started']:.1f}s"
            rec["result"] = _jsonable(summary or {})
            if raw and not error:
                rec["result"].update(_plot_payload(rec["run"], raw))
            rec["error"] = str(error) if error else None
            rec["_progress_arr"] = None

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
                            stage = 1
                        item["progress"] = round(max(0, min(100, frac * 100)), 1)
                        item["stage"] = ("准备" if stage == 0 else "积分" if stage == 1 else "收尾")
                        item["elapsed"] = f"{time.monotonic() - rec['_started']:.1f}s"
                    except Exception:
                        pass
                out.append(item)
            return {"ok": True, "runs": list(reversed(out)), "config": _jsonable(config or {})}


class RunMonitorServer:
    def __init__(self, store, root, config_reader, host="127.0.0.1", port=0, port_file=None):
        self.store, self.root, self.config_reader = store, Path(root), config_reader
        self.host, self.port, self.port_file = host, int(port), Path(port_file) if port_file else None
        self.httpd = None
        self._thread = None

    def start(self):
        if self.httpd is not None:
            return self
        store, root, config_reader = self.store, self.root.resolve(), self.config_reader
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return
            def _send(self, code, payload, content_type="application/json; charset=utf-8"):
                data = payload if isinstance(payload, bytes) else (json.dumps(_jsonable(payload), ensure_ascii=False, allow_nan=False).encode() if not isinstance(payload, str) else payload.encode())
                self.send_response(code); self.send_header("Content-Type", content_type); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            def do_GET(self):
                path = unquote(urlparse(self.path).path)
                if path == "/api/runs": self._send(200, store.snapshot(config_reader())); return
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
