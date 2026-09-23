# -*- coding: utf-8 -*-
"""驾驶舱自检（Phase 3）——独立只读投影的 5 条验收（含致死性证明）。

  A ① 黑匣子独立存活（T5）：写事件到 journal → **丢弃 Bus（模拟项目进程死）** →
     驾驶舱后端仍从**磁盘文件**读回死前全史 + 最后状态。
  B ② diff-since-seq 增量：连续读只回新事件，不重灌全量（R1 头号原语）。
  C ③ 告警灯致死性（M5/D4）：构造一次"静默降级"（runspec.warnings 非空）→ 灯**必须**
     亮；干净运行 → **不亮**（否则是永不触发的哑告警，视为 FAIL）；失败/错误 → error。
  D ④ 只读（I2）：AST 守卫证明 src/cockpit 无任何写回通道（不 import mecha 写侧、
     无写模式 open、无破坏性 os/shutil）；**致死性**：内联裸写片段必被检出。
  E ⑤ 远程只读 + 鉴权 + 不暴露写端点（T6）：真起 uvicorn + urllib——无 token→401、
     有 token→200、POST→405（只读）、静态前端可达。

用临时 root（不污染 `<repo>/.mecha`）。
"""
import ast
import gc
import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

for _n in ('uvicorn', 'uvicorn.error', 'uvicorn.access', 'asyncio'):
    import logging
    logging.getLogger(_n).setLevel(logging.ERROR)

from checks.checklib import Checker                        # noqa: E402
from mecha.bus import Bus                             # noqa: E402
from cockpit import backend                           # noqa: E402
from cockpit import server as CSVR                    # noqa: E402

C = Checker('cockpit')
import cockpit as _cockpit_pkg        # 独立库 mecha-sdk（editable 安装）
_COCKPIT = os.path.dirname(os.path.abspath(_cockpit_pkg.__file__))

# ---- 建一个有事件的 journal，然后"杀掉项目"（丢弃 Bus 对象）----
_TMP = tempfile.mkdtemp(prefix='cockpit_chk_')
_bus = Bus(root=_TMP)
# P0-1：Bus(root=X) 的 X 是**项目根**，journal 落 `X/.mecha`（不再是 X 本身）。
# 驾驶舱读的是「数据根」，所以这里统一用 bus.root。
_TMP = _bus.root
_bus.append(actor='ai', kind='set', target='field.B_gauss', before=None,
            after=21.7, reason='Zeeman 调谐', provenance='session.set',
            reversibility='undo')
_bus.append(actor='ai', kind='run', target='sim:mixed:single',
            after={'ok': True, 'n_states': 4}, reason='首跑',
            provenance={'config': {'dynamic.method': 'mixed'},
                        'runspec': {'warnings': []}}, reversibility='rerun')
_LAST = _bus.last_seq
del _bus
gc.collect()                       # ← 项目进程"死了"，只剩磁盘 journal

# ---------------- A ① 黑匣子独立存活 ----------------
evs = backend.read_events(_TMP, 0)
ori = backend.orientation(_TMP)
C.group('A ① 独立进程：杀项目后仍读到死前全史 + 最后状态（T5 黑匣子）', [
    ('Bus 已丢弃，journal 文件仍在盘上',
     os.path.exists(os.path.join(_TMP, 'journal.jsonl'))),
    ('从磁盘读回全部 %d 条（== 死前 last_seq %d）' % (len(evs), _LAST),
     len(evs) == _LAST),
    ('最后状态可读（最后一次 set 的 after=21.7）',
     any(e['kind'] == 'set' and e['after'] == 21.7 for e in evs)),
    ('orientation 定位包给出 last_seq/n_events/n_runs',
     ori['last_seq'] == _LAST and ori['n_events'] == len(evs)
     and ori['n_runs'] == 1),
])

# ---------------- B ② diff-since-seq ----------------
inc_all = backend.read_events(_TMP, 0)
inc_1 = backend.read_events(_TMP, 1)
C.group('B ② diff-since-seq：连续读只回新事件，不重灌全量（R1）', [
    ('since=0 回全部 %d 条' % len(inc_all), len(inc_all) == _LAST),
    ('since=1 只回 seq>1（增量 %d 条）' % len(inc_1),
     len(inc_1) == _LAST - 1 and all(e['seq'] > 1 for e in inc_1)),
    ('since=last 回 0 条（无新事件不重灌全量）',
     backend.read_events(_TMP, _LAST) == []),
])

# ---------------- C ③ 告警致死性（告警内容由适配器/ops 提供；cockpit 透传+兜底） ----------------
deg_alert = {'seq': 900, 'kind': 'alert', 'actor': 'ai',
             'provenance': {'level': 'warn'}, 'after': {'rule': 'degradation', 'n': 1},
             'reason': '引擎改道：测试降级'}
ovl_alert = {'seq': 904, 'kind': 'alert', 'actor': 'ai',
             'provenance': {'level': 'warn'}, 'after': {'rule': 'overlap'},
             'reason': '同一跃迁有 2 条时间重叠驱动——扫描忘了清空'}
clean = {'seq': 901, 'kind': 'run', 'actor': 'ai', 'after': {'ok': True}}
failed = {'seq': 902, 'kind': 'run', 'after': {'ok': False, 'msg': '引擎报错'}}
err_ev = {'seq': 903, 'kind': 'error', 'reason': '契约违反', 'actor': 'engine'}
a_deg = backend.evaluate_alerts([deg_alert])
a_ovl = backend.evaluate_alerts([ovl_alert])
a_clean = backend.evaluate_alerts([clean])
a_fail = backend.evaluate_alerts([failed])
a_err = backend.evaluate_alerts([err_ev])


# 产出侧致死性：ops 见 check_run warnings 必落 kind='alert' 事件（领域判断→事件的桥）
class _WarnEngine:
    def engine_params(self): return {}
    def _rec(self, n, f, d, lo, hi):
        return {'name': n, 'type': 'float', 'unit': '', 'lo': lo, 'hi': hi,
                'desc': 't', 'default': d, 'family': f, 'readonly': False}
    def config_schema(self): return {'x': self._rec('x', 'config', 0.0, -1, 1)}
    def solve_schema(self): return {}
    def solve_map(self): return {}
    def settable_keys(self): return ['x']
    def validate_wave_list(self, v): return v
    def make_cache(self): return object()
    def build_run(self, cache, snap, diag=False):
        return {'method': 'warn', 'two_body': False, 'spec': dict(snap)}
    def check_run(self, run): return ['测试：静默降级警告']
    def evolve_run(self, run, progress_arr=None): return {'ok': True}
    def result_summary(self, run, res, **kw): return {'ok': True}
    def summarize_run(self, run): return 'warn run'
    def estimate(self, cache, snap): return {'sec': 0.0}
    def preview(self, cache, snap): return {'ok': True}
    def list_levels(self, cache, snap): return []
    def run_annotations(self, snap): return {'n_waves': 0, 'overlap_warning': None}


from mecha import runtime as _rt                     # noqa: E402
from mecha.ops import Ops as _Ops                    # noqa: E402
from mecha.session import Session as _Session        # noqa: E402
_TMP2 = tempfile.mkdtemp(prefix='cockpit_warn_')
_rt.set_engine(_WarnEngine())
_wsess = _Session(root=_TMP2)
_wops = _Ops(_wsess)
_wops.run(actor='test', reason='warn run', land_artifact=False)
a_ops = backend.evaluate_alerts(backend.read_events(_wsess.bus.root, 0))
C.group('C ③ 告警灯致死性：适配器/ops 落 alert 事件→cockpit 透传亮；干净→不亮（防哑告警 D4）', [
    ('**致死性(产出侧)**：ops 见 check_run warnings → 落 alert 事件 → 透传亮（rule=degradation）',
     any(x['rule'] == 'degradation' for x in a_ops)),
    ('透传：alert 事件(degradation) → 亮且 level=warn',
     len(a_deg) >= 1 and a_deg[0]['rule'] == 'degradation'
     and backend.top_level(a_deg) == 'warn'),
    ('透传：overlap alert → warn',
     any(x['rule'] == 'overlap' for x in a_ovl)
     and backend.top_level(a_ovl) == 'warn'),
    ('干净运行（无 alert 事件）→ 不亮（不是永远亮的哑告警）',
     a_clean == [] and backend.top_level(a_clean) == 'ok'),
    ('run 失败（after.ok=False）→ error 兜底',
     any(x['rule'] == 'run_failed' for x in a_fail)
     and backend.top_level(a_fail) == 'error'),
    ('error 事件 → error 兜底',
     any(x['rule'] == 'error_event' for x in a_err)),
])


# ---------------- D ④ 只读 AST 守卫 ----------------
def _scan_writes(path):
    """扫一个 .py：返回写回/破坏性操作 [(kind, lineno, detail)]。"""
    tree = ast.parse(open(path, encoding='utf-8').read())
    hits = []
    for nd in ast.walk(tree):
        if isinstance(nd, ast.ImportFrom) and nd.module and 'mecha' in nd.module:
            hits.append(('import-mecha', nd.lineno, nd.module))
        if isinstance(nd, ast.Import):
            for al in nd.names:
                if 'mecha' in al.name:
                    hits.append(('import-mecha', nd.lineno, al.name))
        if (isinstance(nd, ast.Call) and isinstance(nd.func, ast.Name)
                and nd.func.id == 'open'):
            mode = None
            if len(nd.args) >= 2 and isinstance(nd.args[1], ast.Constant):
                mode = nd.args[1].value
            for kw in nd.keywords:
                if kw.arg == 'mode' and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            if mode and any(c in str(mode) for c in 'wax+'):
                hits.append(('open-write', nd.lineno, str(mode)))
        if isinstance(nd, ast.Call) and isinstance(nd.func, ast.Attribute):
            v = nd.func.value
            base = v.id if isinstance(v, ast.Name) else ''
            if base in ('os', 'shutil') and nd.func.attr in (
                    'remove', 'unlink', 'replace', 'rename', 'rmtree',
                    'removedirs', 'truncate'):
                hits.append(('destructive', nd.lineno,
                             '%s.%s' % (base, nd.func.attr)))
            if nd.func.attr in ('write_text', 'write_bytes', 'writelines'):
                hits.append(('file-write', nd.lineno, nd.func.attr))
    return hits


viol = []
for fn in sorted(os.listdir(_COCKPIT)):
    if fn.endswith('.py'):
        for h in _scan_writes(os.path.join(_COCKPIT, fn)):
            viol.append('%s:%s:%d(%s)' % (fn, h[0], h[1], h[2]))
# 致死性：三段裸写片段都必须被检出（否则守卫是永不触发的哑守卫）
_ROGUE = [
    "from mecha.session import Session\nSession().set('x', 1, actor='a', reason='r')\n",
    "def f(p):\n    open(p, 'a').write('x')\n",
    "import os\ndef g(p):\n    os.remove(p)\n",
]


def _scan_src(src):
    import tempfile as _tf
    p = os.path.join(_tf.gettempdir(), '_rogue_cockpit_%d.py' % abs(hash(src)))
    with open(p, 'w', encoding='utf-8') as f:      # 仅扫描用，落在系统 temp
        f.write(src)
    try:
        return _scan_writes(p)
    finally:
        os.remove(p)


lethal_hits = [len(_scan_src(s)) for s in _ROGUE]
C.group('D ④ 驾驶舱只读：无写回通道（AST 守卫 + 致死性）', [
    ('src/cockpit/*.py 无写回/破坏性操作（违规：%s）' % (viol or '无'), not viol),
    ('**致死性**：import mecha 写侧 被检出', lethal_hits[0] >= 1),
    ('**致死性**：写模式 open 被检出', lethal_hits[1] >= 1),
    ('**致死性**：os.remove 被检出', lethal_hits[2] >= 1),
])


# ---------------- E ⑤ 远程只读 + 鉴权 + 不暴露写端点 ----------------
def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _http(method, url, data=None):
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            return r.status, r.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8', 'replace')
    except Exception as e:                          # noqa: BLE001
        return -1, '%s: %s' % (type(e).__name__, e)


def _serve(app, port):
    import uvicorn
    cfg = uvicorn.Config(app, host='127.0.0.1', port=port, log_level='error')
    srv = uvicorn.Server(cfg)
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(80):
        try:
            with socket.create_connection(('127.0.0.1', port), 0.2):
                return srv
        except OSError:
            time.sleep(0.1)
    return srv


_app = CSVR.create_app(root=_TMP, token='sekret')
_port = _free_port()
_srv = _serve(_app, _port)
_base = 'http://127.0.0.1:%d' % _port
try:
    no_auth = _http('GET', _base + '/api/history')
    with_auth = _http('GET', _base + '/api/history?token=sekret&since_seq=0')
    posted = _http('POST', _base + '/api/history?token=sekret', data=b'{}')
    health = _http('GET', _base + '/healthz?token=sekret')
    index = _http('GET', _base + '/')
    alerts = _http('GET', _base + '/api/alerts?token=sekret')
    # ⚠ 必须在 `finally` 关服务**之前**取（否则请求打不通）
    branding = _http('GET', _base + '/api/branding?token=sekret')
finally:
    _srv.should_exit = True
    time.sleep(0.3)

try:
    wa = json.loads(with_auth[1])
except Exception:                                    # noqa: BLE001
    wa = {}
try:
    hl = json.loads(health[1])
except Exception:                                    # noqa: BLE001
    hl = {}
C.group('E ⑤ 远程只读（鉴权）+ 不暴露写端点（T6）', [
    ('无 token → 401（远程只读**必须**鉴权）', no_auth[0] == 401),
    ('有 token → 200 且读回事件（另一台设备浏览器能看）',
     with_auth[0] == 200 and wa.get('count', 0) == _LAST),
    ('POST 写请求 → 405（只读，无写端点）', posted[0] == 405),
    ('healthz 自报 read_only=True', health[0] == 200 and hl.get('read_only') is True),
    ('静态前端可达（GET / → 200，含 HTML 标题）',
     # ⚠ 断言**不钉任何品牌串**（W1/D-1 把出厂标题中性化了：旧版是 "Re0 机甲驾驶舱"）。
     # 这里只验"前端真能取到、是个带 <title> 的 HTML"——品牌由项目经 /api/branding 注入。
     index[0] == 200 and '<title>' in index[1]),
    ('品牌注入端点在位（/api/branding 可读，且**不含项目名**）',
     branding[0] == 200 and 'Re0' not in branding[1]
     and 'title' in branding[1]),
    ('alerts 端点给出级别（平静 ok）',
     alerts[0] == 200 and json.loads(alerts[1]).get('level') == 'ok'),
])

sys.exit(0 if C.summary() else 1)
