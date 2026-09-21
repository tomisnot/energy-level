# -*- coding: utf-8 -*-
"""GUI 的 Hub-client 门面（与 AppBridge 同形，但经 HubClient/MCP 连 Hub）。

Phase 1（GUI Hub 客户端化）：GUI 不再自托管内核，而是连**已在跑的 Hub**，与 AI 操作
**同一个 session**。本门面把 AppBridge 的方法面映射到 MCP 工具：

    read/effective  → get_state          params/describe → list_params
    write           → set_param          write_many      → mecha_script(set_many)
    undo            → undo               run/estimate/preview → run_compute/estimate_cost/preview_behavior
    activity        → get_activity       subscribe       → 轮询 get_state 差分
    save/load       → get_state 读回本地 / 读本地后 write_many 回 Hub

**switch_mode 不经 MCP**（AI 不能给自己授权）：人工夺权走"起 GUI 连 Hub + 人工控制端点"
（Phase 1 后续）；此处 switch_mode 抛 HubError 指明路径，不静默。
"""
import json
import os
import threading
import time

from .hub_client import HubClient, HubError

__all__ = ['HubBridge']


class HubBridge:
    """连 Hub 的 GUI 门面。method 面与 AppBridge 对齐，GUI 代码可无感切换 transport。"""

    def __init__(self, url, timeout=60.0):
        self.client = HubClient(url, timeout=timeout)
        self.client.initialize()
        self._stop = threading.Event()

    # ---------------- 与 AppBridge 对齐的附加属性（GUI 无感切换 transport） ----------------
    @property
    def cache(self):
        """引擎缓存在 Hub 侧；GUI 经 list_levels/get_state 取态，本地无 cache。"""
        return None

    @property
    def fields(self):
        """建场层经 MCP（set_drive/get_waves）；本地无 fields 对象。"""
        return None

    @property
    def session(self):
        """无本地 session；单一真值源在 Hub。"""
        return None

    @property
    def ops(self):
        """run/estimate/preview 已在本门面；prepare/execute 不适用（Hub 侧 run）。"""
        return self

    # ---------------- 只读投影 ----------------
    def _state(self):
        return self.client.call('get_state') or {}

    def read(self, key):
        return (self._state().get('params') or {}).get(key)

    def effective(self):
        return self._state().get('params') or {}

    def params(self, **kw):
        return self.client.call('list_params', **{
            k: v for k, v in kw.items() if v is not None})

    def describe(self, key):
        rec = self.client.call('list_params', prefix=key) or {}
        for p in (rec.get('params') or []):
            if p.get('name') == key:
                return p
        raise HubError('describe: 找不到键 %r' % key)

    def activity(self, since_seq=0):
        return self.client.call('get_activity') or {}

    def orientation(self, since_seq=0):
        return self.client.call('get_state') or {}

    @property
    def mode(self):
        return (self._state().get('mode')) or 'human'

    def can_write(self, actor='gui'):
        return self.mode == 'human'

    def subscribe(self, cb, interval=1.0):
        """轮询 get_state 差分 → cb(changed_keys)。返回取消函数。"""
        last = dict(self.effective())

        def _loop():
            while not self._stop.is_set():
                time.sleep(interval)
                if self._stop.is_set():
                    return
                try:
                    cur = self.effective()
                except Exception:                          # noqa: BLE001
                    continue
                changed = [k for k in set(cur) | set(last) if cur.get(k) != last.get(k)]
                last.clear()
                last.update(cur)
                if changed:
                    cb(changed)
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        return self._stop.set

    # ---------------- 写（经 Hub choke point） ----------------
    def write(self, key, value, reason, actor='gui'):
        return self.client.call('set_param', key=key, value=value, reason=reason)

    def write_many(self, patch, reason, actor='gui'):
        script = 'result = session.set_many(%r, %r)' % (patch, reason)
        return self.client.call('mecha_script', script=script)

    def undo(self, reason='undo', actor='gui'):
        return self.client.call('undo', reason=reason)

    # ---------------- 运行 / 预览 ----------------
    def run(self, reason, actor='gui', diag=False, progress_arr=None):
        return self.client.call('run_compute', reason=reason, diag=bool(diag))

    def estimate(self):
        return self.client.call('estimate_cost')

    def preview(self):
        return self.client.call('preview_behavior')

    def in_flight(self):
        jobs = (self.client.call('list_jobs') or {}).get('jobs') or []
        return sum(1 for j in jobs if j.get('status') == 'running')

    def switch_mode(self, new_mode, reason, actor='human'):
        """人工夺权：走 Hub 的人工控制端点（token 鉴权，**非 MCP**——AI 不能自授权）。"""
        import urllib.request
        try:
            with open('.human-token', encoding='utf-8') as f:
                token = f.read().strip()
        except OSError:
            raise HubError('找不到 .human-token（Hub 未起人工控制端点）')
        base = self.client.url.replace('/mcp', '')
        url = base.rsplit(':', 1)[0] + ':8767/switch_mode'   # 人工端点固定 8767
        req = urllib.request.Request(
            url, data=json.dumps({'token': token, 'mode': new_mode,
                                  'reason': reason}).encode('utf-8'),
            headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=self.client.timeout) as r:
            body = json.loads(r.read().decode('utf-8'))
        if not body.get('ok'):
            raise HubError('switch_mode 失败：%s' % body.get('error'))
        return body

    # ---------------- 存档（本地文件 ↔ Hub session） ----------------
    def save(self, path, reason='save'):
        snap = {'format': 'mecha-session-v1', 'mode': self.mode,
                'params': self.effective()}
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(snap, f, ensure_ascii=False, indent=1, default=str)
        return {'ok': True, 'path': path, 'n_keys': len(snap['params'])}

    def load(self, path, reason='load', actor='gui'):
        with open(path, encoding='utf-8') as f:
            snap = json.load(f)
        if not isinstance(snap, dict) or 'params' not in snap:
            raise ValueError('不是 mecha-session-v1 存档（缺 params）：%r' % path)
        return self.write_many(snap['params'] or {}, '%s: %s' % (reason, path))

    def close(self):
        self._stop.set()
        self.client.close()
