# -*- coding: utf-8 -*-
"""数据回收页自检（P4-3）——offscreen 验"列出 + 勾选 + 触发 + 回执"这条接线，
并钉死一条纪律：**GUI 不得自己删文件**（删除逻辑一律在库侧 `mecha.gc`）。

覆盖：
  A. 装配：检视器第 7 个 tab「存档/产物」在位，**既有 6 页名/序不变**（M3）；
     本机数据根 = `<repo>/.mecha`。
  B. 干跑优先：干跑走**库侧** `mecha.gc.gc(dry_run=True)`（spy 验证），
     且真跑按钮在"看过计划且有可删项"之前**不可用**。
  C. 二次确认：`_gc_confirm()` 返回 False → **库侧一次都不被调用**（不删）；
     返回 True → 以 `dry_run=False` 调库侧。
  D. 归档 ≠ 删除：三份清单同表分列，动作列明确区分
     「删除」/「归档（移动，不删）」/「保留」，且保留项显示 `why`。
  E. **GUI 不自己删**（源码级反向断言）：`src/ui/*.py` + `app_entry.py` 里
     `.remove/.rmtree/.unlink/.rmdir/.removedirs` 属性调用 **0 命中**；
     并带**扫描器自身的致死性**（合成一段含 `shutil.rmtree` 的源码必须被检出）。
  F. 致死性（接线）：把库侧改成抛错 → 页面必须显示"回收失败"（不许静默当成功）。

⚠ 本判据**不真删任何东西**：B/D 用 spy 或合成报告；C 的"真跑"分支用 spy 拦截，
  不会到达库侧的删除代码。
offscreen；os._exit 绕过 teardown 噪声。
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')

from PyQt5.QtWidgets import QApplication                  # noqa: E402
from checks.checklib import Checker                        # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)
C = Checker('gui_gc')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import app_entry                                          # noqa: E402
from mecha import gc as _GC                               # noqa: E402

_w = app_entry.MainWindow()
_d = _w.dynamic
_d.set_god_mode(True)
_insp = _d._inspector

# ---------------- A. 装配（新 tab + 既有页不动） ----------------
_names = [_insp.tabs.tabText(i) for i in range(_insp.tabs.count())]
_root = _insp._gc_root()
C.group('A 回收页装配（第 7 个 tab；既有 6 页不动 = M3）', [
    ('tab 数为 7 且第 7 个是「存档/产物」（%s）' % _names,
     len(_names) == 7 and _names[6] == '存档/产物'),
    ('既有 6 页名与序不变（M3）',
     _names[:6] == ['处理链', '哈密顿量', '耦合对', '求解过程', '扇区与重构', '引擎参数']),
    ('本机数据根 = <repo>/.mecha（%s）' % _root,
     _root is not None and os.path.normpath(_root)
     == os.path.normpath(os.path.join(ROOT, '.mecha'))),
    ('真跑按钮初始不可用（先干跑看过计划）', _insp.gc_real_btn.isEnabled() is False),
])

# ---------------- B. 干跑优先：走库侧 + dry_run=True ----------------
_calls = []
_real_gc = _GC.gc


def _spy(*a, **k):
    # 位置参数也要收：`_GcWorker` 把 root 当**位置参数**传（判据第一版只收 kwargs，
    # 于是"传了数据根"那条假红——这是断言写窄了，不是代码坏）。
    _calls.append({'args': a, 'kwargs': k})
    return {'ok': True, 'dry_run': bool(k.get('dry_run')), 'plan': [], 'kept': [],
            'archive': [], 'freed_bytes': 0, 'archived_bytes': 0, 'notes': [],
            'note': '（判据 spy）'}


_GC.gc = _spy
try:
    _insp._gc_start_dry()
    _insp._gc_worker.wait(30000)
    app.processEvents()
finally:
    _GC.gc = _real_gc
_dry_call = _calls[-1] if _calls else {}
_dry_root = (_dry_call.get('args') or (None,))[0]
C.group('B 干跑优先：经库侧且 dry_run=True（GUI 不自算、不自删）', [
    ('干跑调用了库侧 mecha.gc.gc（%d 次）' % len(_calls), len(_calls) >= 1),
    ('传 dry_run=True（%r）' % _dry_call.get('kwargs', {}).get('dry_run'),
     _dry_call.get('kwargs', {}).get('dry_run') is True),
    ('传了本机数据根（%s）' % _dry_root,
     str(_dry_root or '').endswith('.mecha')),
    ('传了 bus（回收要留 kind="gc" 事件）', 'bus' in _dry_call.get('kwargs', {})),
    ('传了触发者（审计链：谁做的）',
     bool(_dry_call.get('kwargs', {}).get('triggered_by'))),
])

# ---------------- C. 真跑必须二次确认 ----------------
_calls.clear()
_orig_confirm = type(_insp)._gc_confirm
_GC.gc = _spy
try:
    type(_insp)._gc_confirm = lambda self: False        # 人点了"取消"
    _insp._gc_start_real()
    _n_cancel = len(_calls)
    _cancel_receipt = _insp.gc_receipt.text()           # 就地取（下一步会覆盖它）
    type(_insp)._gc_confirm = lambda self: True         # 人确认
    _insp._gc_start_real()
    _insp._gc_worker.wait(30000)
    app.processEvents()
    _real_call = _calls[-1] if _calls else {}
    _n_confirmed = len(_calls)
finally:
    type(_insp)._gc_confirm = _orig_confirm
    _GC.gc = _real_gc
C.group('C 真跑必须人的二次确认（取消 → 库侧零调用）', [
    ('确认返回 False 时库侧**一次都没被调用**（%d 次）' % _n_cancel,
     _n_cancel == 0),
    ('取消时页面明说未执行（%s）' % _cancel_receipt[:24],
     '已取消' in _cancel_receipt or '未执行' in _cancel_receipt),
    ('确认后调用库侧且 dry_run=False（%r）'
     % _real_call.get('kwargs', {}).get('dry_run'),
     _n_confirmed == 1 and _real_call.get('kwargs', {}).get('dry_run') is False),
])

# ---------------- D. 归档 ≠ 删除（同表分列 + 动作语义分明） ----------------
_fake = {
    'ok': True, 'dry_run': True,
    'plan': [{'area': 'artifacts', 'aid': 'a1', 'path': 'artifacts/a1',
              'bytes': 2048, 'reason': '未 pinned、超 90 天、且不在最新 50 个内'},
             {'area': 'payloads', 'file': 'p1.npz', 'path': 'payloads/p1.npz',
              'bytes': 1024, 'reason': '三源（journal∪artifacts∪jobs）均无引用'}],
    'archive': [{'from': 'journal.20260101-000000.jsonl',
                 'to': 'archive/journal.20260101-000000.jsonl', 'bytes': 4096,
                 'reason': 'journal 轮转段超 90 天 → **归档**（移动，不删除）'}],
    'kept': [{'area': 'journal', 'why': '唯一真值日志（I2）——只归档、不删除'},
             {'area': 'artifacts', 'aid': 'a2', 'why': 'pinned'},
             {'area': 'jobs', 'id': 'j1', 'why': 'running —— **永不删**（诚实对账要保留）'}],
    'freed_bytes': 3072, 'archived_bytes': 4096, 'notes': [],
    'note': '干跑：未删除任何东西。',
}
_insp._gc_fill_table(_fake)
_acts = [(_insp.gc_table.item(r, 0).text(), _insp.gc_table.item(r, 1).text(),
          _insp.gc_table.item(r, 2).text()) for r in range(_insp.gc_table.rowCount())]
_del = [a for a in _acts if a[0] == '删除']
_arc = [a for a in _acts if a[0].startswith('归档')]
_keep = [a for a in _acts if a[0] == '保留']
_keep_why = [_insp.gc_table.item(r, 4).text()
             for r in range(_insp.gc_table.rowCount())
             if _insp.gc_table.item(r, 0).text() == '保留']
C.group('D 归档≠删除：三份清单同表分列，动作语义分明', [
    ('行数 = plan+archive+kept = %d' % _insp.gc_table.rowCount(),
     _insp.gc_table.rowCount() == 6),
    ('删除类 %d 行（artifacts + payloads）' % len(_del), len(_del) == 2),
    ('归档类 %d 行且动作文字含"移动，不删"（%s）'
     % (len(_arc), _arc[0][0] if _arc else ''),
     len(_arc) == 1 and '不删' in _arc[0][0]),
    ('归档行显示 from → to（%s）' % (_arc[0][2] if _arc else ''),
     bool(_arc) and '→' in _arc[0][2] and 'archive/' in _arc[0][2]),
    ('保留类 %d 行且显示 why（%s）' % (len(_keep), _keep_why),
     len(_keep) == 3 and any('pinned' in w for w in _keep_why)
     and any('永不删' in w for w in _keep_why)),
])

# ---------------- E. GUI 不自己删（源码级反向断言 + 扫描器致死性） ----------------
_BAD_ATTRS = {'remove', 'rmtree', 'unlink', 'rmdir', 'removedirs'}
# **只算"删文件"那一类**：接收者是 os / shutil / pathlib / Path(...) / QFile / QDir。
# ⚠ 判据第一版只看属性名 ⇒ 把 `artist.remove()` 误判成删文件（`ui_radar.py:153,295`
# 是 matplotlib artist 的内存移除，不是文件删除）——**断言写宽了会假红**，同样要修。
_FILE_RECV = ('os', 'shutil', 'pathlib', 'QFile', 'QDir')


def _scan(src_text, label):
    """AST 扫"删文件"调用（注释/字符串天然不算——AST 看不见它们）。"""
    hits = []
    try:
        tree = ast.parse(src_text)
    except SyntaxError as e:
        return ['%s: SyntaxError %s' % (label, e)]
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr in _BAD_ATTRS):
            continue
        try:
            recv = ast.unparse(node.value)
        except Exception:                                   # noqa: BLE001
            recv = ''
        head = recv.split('.')[0].split('(')[0].strip()
        if head in _FILE_RECV or 'Path(' in recv:
            hits.append('%s:%d %s.%s' % (label, getattr(node, 'lineno', 0),
                                         recv, node.attr))
    return hits


_hits = []
for _fn in sorted(os.listdir(os.path.join(ROOT, 'src', 'ui'))):
    if _fn.endswith('.py'):
        _p = os.path.join(ROOT, 'src', 'ui', _fn)
        _hits += _scan(open(_p, encoding='utf-8').read(), 'src/ui/' + _fn)
_hits += _scan(open(os.path.join(ROOT, 'app_entry.py'), encoding='utf-8').read(),
               'app_entry.py')
# 扫描器自身的致死性：合成一段"GUI 自己删"的源码必须被检出
_lethal_src = ('import os, shutil\n'
               'def bad(p):\n    os.remove(p)\n    shutil.rmtree(p)\n')
_lethal_hits = _scan(_lethal_src, '<synthetic>')
# 反向致死性：**内存里的 artist.remove() 不得被误判**（否则判据会假红）
_mem_src = ('def ok(a):\n    a.remove()\n    self._cb.remove()\n')
_mem_hits = _scan(_mem_src, '<synthetic-mem>')
C.group('E GUI 不自己删文件（删除逻辑一律在库侧 mecha.gc）', [
    ('src/ui/*.py + app_entry.py 里"删文件"调用 0 命中（%s）' % (_hits or '无'),
     not _hits),
    ('回收页只调库侧：源码含 mecha.gc 调用（%s）'
     % ('有' if 'mecha.gc' in open(os.path.join(
         ROOT, 'src', 'ui', 'solver_inspector.py'), encoding='utf-8').read() else '无'),
     'mecha.gc' in open(os.path.join(
         ROOT, 'src', 'ui', 'solver_inspector.py'), encoding='utf-8').read()),
    ('致死性：扫描器能检出合成源码里的 os.remove/shutil.rmtree（%d 处）'
     % len(_lethal_hits), len(_lethal_hits) >= 2),
    ('反向致死性：内存里的 artist.remove() **不**被误判（%d 处）'
     % len(_mem_hits), len(_mem_hits) == 0),
])

# ---------------- F. 致死性：库侧抛错必须显示失败（不许静默当成功） ----------------
def _boom(*a, **k):
    raise RuntimeError('判据故意让回收失败')


_GC.gc = _boom
try:
    _insp._gc_start(dry_run=True)
    _insp._gc_worker.wait(30000)
    app.processEvents()
finally:
    _GC.gc = _real_gc
C.group('F 致死性：库侧失败 → 页面显示"失败"（不静默当成功）', [
    ('状态显示失败（%s）' % _insp.gc_sum.text(), '失败' in _insp.gc_sum.text()),
    ('回执含异常信息（%s）' % _insp.gc_receipt.text()[:40],
     '回收失败' in _insp.gc_receipt.text() and '故意让回收失败' in _insp.gc_receipt.text()),
    ('按钮已复位（干跑可再点）', _insp.gc_dry_btn.isEnabled() is True),
])

_ok = C.summary()
sys.stdout.flush()
os._exit(0 if _ok else 1)
