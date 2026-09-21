# -*- coding: utf-8 -*-
"""async job：长任务可发起、可轮询、**断线可找回**（job 落盘）。

动机（纲领 R2「回合数」/「context」）：两体演化动辄分钟级，不能让 AI 干等一个同步
回程。作业化后：AI 发起 → 拿 job_id → 轮询进度/结果 → **断线重连仍按 id 取回**
（job 状态落 `.mecha/jobs/<id>.json`，与进程存活无关）。

- 进度用 `[阶段内进度, 取消标志, 阶段]` 三槽——**对齐 `evolve_run` 的 progress_arr**
  （写 [1]=1 即请求取消，引擎在步间复核）。
- 每次运行仍走 `ops.run` → `runspec` 唯一分派（I5 人机同路径）；job 只是"异步地调它"。
- 进程重启后，遗留在 'running' 但已无活线程的 job 被标为 'interrupted'（诚实对账，
  不假装还在跑）。
"""
import json
import os
import threading
import time

import numpy as np

from .errors import MechaError

__all__ = ['JobRunner']


class JobRunner:
    def __init__(self, ops, root=None):
        self.ops = ops
        self.session = ops.session
        self.root = root or ops.root
        self.jobs_dir = os.path.join(self.root, 'jobs')
        os.makedirs(self.jobs_dir, exist_ok=True)
        self._live = {}                     # jid -> {'prog': np.ndarray(3)}
        self._lock = threading.Lock()
        self._reconcile()                   # 重启对账：孤儿 running → interrupted

    # ---------------- 落盘（可找回） ----------------
    def _path(self, jid):
        return os.path.join(self.jobs_dir, '%s.json' % jid)

    def _write(self, jid, data):
        with self._lock:
            tmp = self._path(jid) + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, default=str)
            os.replace(tmp, self._path(jid))

    def _read(self, jid):
        try:
            with open(self._path(jid), encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _reconcile(self):
        for j in self._all_raw():
            if j.get('status') == 'running' and j['id'] not in self._live:
                j['status'] = 'interrupted'
                j['updated'] = time.time()
                j['error'] = {'kind': 'interrupted',
                              'message': '上个进程结束时该 job 仍在跑（断线可重发）'}
                self._write(j['id'], j)

    def _all_raw(self):
        out = []
        try:
            names = sorted(os.listdir(self.jobs_dir))
        except OSError:
            return out
        for fn in names:
            if fn.endswith('.json'):
                j = self._read(fn[:-5])
                if j:
                    out.append(j)
        return out

    def _overlay(self, job):
        """把活线程的实时进度叠到 job（disk 只在起止写，进度走内存）。"""
        live = self._live.get(job.get('id'))
        if live is not None:
            p = live['prog']
            job['progress'] = [round(float(p[0]), 4), int(p[1]), int(p[2])]
        return job

    # ---------------- 发起 ----------------
    def submit(self, *, reason, actor='ai', config=None, kind='run'):
        """发起一个后台 job → 返回 job_id。config 给了就先经 choke point 写进会话。"""
        from .ops import config_id
        if not reason:
            raise MechaError('submit 需要 reason（M1 归因）')
        if config:
            self.session.set_many(config, actor=actor,
                                  reason='job 配置：%s' % reason)
        snap = self.session.effective()
        jid = '%d-%s' % (int(time.time() * 1000), config_id(snap)[:6])
        prog = np.zeros(3)                   # [frac, cancel, stage]
        job = {'id': jid, 'kind': kind, 'status': 'running', 'actor': actor,
               'reason': reason, 'config': snap, 'created': time.time(),
               'updated': time.time(), 'progress': [0.0, 0, 0],
               'result': None, 'error': None}
        self._write(jid, job)
        self._live[jid] = {'prog': prog}
        threading.Thread(target=self._work, args=(jid, reason, actor, kind, prog),
                         daemon=True).start()
        self.session.bus.append(
            actor=actor, kind='job_submit', target=jid, before=None,
            after={'kind': kind, 'n_config_keys': len(snap)}, reason=reason,
            provenance='jobs.submit', reversibility='cancel')
        return jid

    def _work(self, jid, reason, actor, kind, prog):
        try:
            if kind != 'run':
                raise MechaError('未知 job kind %r（现只支持 run）' % kind)
            res = self.ops.run(actor='job:%s' % jid, reason=reason,
                               progress_arr=prog)
            result = {'ok': res.get('ok'), 'seq': res.get('seq'),
                      'artifact_id': res.get('artifact_id'),
                      'summary': res.get('summary')}
            status = 'cancelled' if int(prog[1]) else 'done'
            self._finish(jid, status, result, None)
        except Exception as e:                          # noqa: BLE001
            self._finish(jid, 'error', None,
                         {'kind': type(e).__name__, 'message': str(e)})
        finally:
            self._live.pop(jid, None)

    def _finish(self, jid, status, result, error):
        job = self._read(jid) or {'id': jid}
        live = self._live.get(jid)
        job.update(status=status, result=result, error=error,
                   updated=time.time(),
                   progress=([round(float(live['prog'][0]), 4),
                              int(live['prog'][1]), int(live['prog'][2])]
                             if live else job.get('progress')))
        self._write(jid, job)
        self.session.bus.append(
            actor='job', kind='job_done', target=jid, before=None,
            after={'status': status,
                   'ok': (result or {}).get('ok') if result else None,
                   'artifact_id': (result or {}).get('artifact_id') if result else None},
            reason='job 收尾', provenance='jobs._work', reversibility='rerun')

    # ---------------- 轮询 / 找回 / 取消 ----------------
    def poll(self, jid):
        job = self._read(jid)
        if job is None:
            raise MechaError('未知 job %r' % jid,
                             hint='用 list_jobs() 看在册 job（断线重连按 id 取回）',
                             suggest=[j['id'] for j in self._all_raw()][-5:])
        return self._overlay(job)

    def list_jobs(self):
        return [self._overlay(j) for j in self._all_raw()]

    def cancel(self, jid):
        """请求取消：写 progress[1]=1（引擎在步间复核后停）。"""
        live = self._live.get(jid)
        if live is None:
            job = self._read(jid)
            if job and job.get('status') == 'running':
                raise MechaError('job %r 不在本进程（随上个进程结束了）；无法取消' % jid)
            return {'ok': False, 'jid': jid, 'reason': 'not live'}
        live['prog'][1] = 1
        return {'ok': True, 'jid': jid, 'requested': 'cancel'}
