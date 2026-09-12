# -*- coding: utf-8 -*-
"""自检：代码新鲜度（T3 / 待办 A12）——"改了 src/ 没重启"必须**一眼看出**。

由来（分工文档 §六 #1，使用者原话）："AI 无法判断软件是否在跑补丁前的代码，
导致白烧一轮算力"（实测：源码里有 `_H_SNAPSHOT_MAX_MB`，进程里 `hasattr` 为 False）。

覆盖：
  A. 纯函数（造临时树，注入 loaded_at）：新于基准 → stale；旧于基准 → 新鲜；
     per-group mtime（src_core / src_ai / src_ui / app）；n_stale / newest 排序与 stale 标记
  B. 真实仓库：`code_status()` 在 ROOT 上跑得通、键齐、类型对（不依赖 Qt）
  C. 接线：`codeact.run_code` 返回里**必定**有 `stale_code_warning`；
     陈旧时它是字符串（含"重启"），新鲜时是 None
  D. 假阴性回归：`mark_loaded()` 是 first-wins —— 进程内**后**导入本模块
     （或再次调用）**不得**把基准推到编辑之后

用法：python checks/ai_code_freshness_selfcheck.py（毫秒级、无 Qt、无 ARC）
"""
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402
from src.ai import code_freshness as CF                          # noqa: E402

C = Checker('ai_code_freshness')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------
# A. 纯函数：造一棵临时树
# ---------------------------------------------------------------
print('A) 纯函数 code_status（造临时树 + 注入 loaded_at）')
tmp = tempfile.mkdtemp(prefix='cf_chk_')
for rel in ('src/core/a.py', 'src/ai/b.py', 'src/ui/c.py',
            'src/top.py', 'app_entry.py'):
    p = os.path.join(tmp, rel.replace('/', os.sep))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write('x = 1\n')
now = time.time()
OLD, NEW = now - 3600.0, now - 10.0


def _touch(rel, ts):
    os.utime(os.path.join(tmp, rel.replace('/', os.sep)), (ts, ts))


try:
    for rel in ('src/core/a.py', 'src/ai/b.py', 'src/ui/c.py',
                'src/top.py', 'app_entry.py'):
        _touch(rel, OLD)
    d = CF.code_status(root=tmp, loaded_at=now - 60.0, use_cache=False)
    C.check('A1 全部文件都旧于基准 → stale=False',
            d['stale'] is False and d['n_stale'] == 0, str(d['hint']))
    C.check('A2 n_files 数准（5 个 .py）', d['n_files'] == 5, str(d['n_files']))
    C.check('A3 分组 mtime：core/ai/ui/app 都 ≈ OLD',
            all(abs((d[k] or 0) - OLD) < 1 for k in
                ('src_core_mtime', 'src_ai_mtime', 'src_ui_mtime',
                 'app_mtime')), str({k: d[k] for k in
                                     ('src_core_mtime', 'src_ai_mtime',
                                      'src_ui_mtime', 'app_mtime')}))

    _touch('src/core/a.py', NEW)                      # 只改一个核心文件
    d = CF.code_status(root=tmp, loaded_at=now - 60.0, use_cache=False)
    C.check('A4 一个 src/core 文件新于基准 → stale=True，n_stale=1',
            d['stale'] is True and d['n_stale'] == 1, str(d['n_stale']))
    C.check('A5 newest[0] 就是那个文件且带 stale 标记',
            d['newest'][0]['path'] == 'src/core/a.py'
            and d['newest'][0]['stale'] is True, str(d['newest'][0]))
    C.check('A6 src_core_mtime 跟着更新、src_ai_mtime 不变',
            abs(d['src_core_mtime'] - NEW) < 1
            and abs(d['src_ai_mtime'] - OLD) < 1)
    C.check('A7 hint 点名"需要重启"', '重启' in d['hint'], d['hint'][:60])
    C.check('A8 age_s > 0（新了多少秒）', d['age_s'] > 0.0, str(d['age_s']))

    _touch('src/ai/b.py', NEW - 5.0)                  # 第二个（稍旧）
    d = CF.code_status(root=tmp, loaded_at=now - 60.0, use_cache=False)
    C.check('A9 多文件陈旧：n_stale=2 且 newest 按 mtime 降序',
            d['n_stale'] == 2 and d['newest'][0]['path'] == 'src/core/a.py'
            and d['newest'][1]['path'] == 'src/ai/b.py', str(d['newest'][:2]))

    # 假警报回归：改 checks/ 下的文件**不该**算 stale（进程不加载它）
    pchk = os.path.join(tmp, 'checks')
    os.makedirs(pchk, exist_ok=True)
    fchk = os.path.join(pchk, 'zzz_selfcheck.py')
    with open(fchk, 'w', encoding='utf-8') as fh:
        fh.write('x = 1\n')
    for rel in ('src/core/a.py', 'src/ai/b.py'):
        _touch(rel, OLD)
    os.utime(fchk, (NEW, NEW))
    d = CF.code_status(root=tmp, loaded_at=now - 60.0, use_cache=False)
    C.check('A9b 改 checks/ 不算 stale（只扫 src/ + app_entry/main）',
            d['stale'] is False and d['n_stale'] == 0
            and all('checks/' not in it['path'] for it in d['newest']),
            str([it['path'] for it in d['newest']]))

    # 基准晚于全部文件 → 新鲜（"重启之后"的形态）
    d = CF.code_status(root=tmp, loaded_at=now + 60.0, use_cache=False)
    C.check('A10 基准晚于全部文件（= 刚重启）→ stale=False',
            d['stale'] is False, str(d['hint']))
    _touch('src/core/a.py', NEW)                      # 重新造一个陈旧文件
    C.check('A11 stale_warning()：陈旧 → 字符串，新鲜 → None',
            isinstance(CF.stale_warning(root=tmp, loaded_at=now - 60.0), str)
            and CF.stale_warning(root=tmp, loaded_at=now + 60.0) is None)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------
# B. 真实仓库（不依赖 Qt）
# ---------------------------------------------------------------
print('B) 真实仓库 code_status()')
d = CF.code_status(use_cache=False)
for k in ('loaded_at', 'stale', 'n_stale', 'n_files', 'src_core_mtime',
          'src_ai_mtime', 'newest', 'hint'):
    C.check('B1 键齐：%s' % k, k in d)
C.check('B2 src_core_mtime / src_ai_mtime 是有限浮点',
        all(isinstance(d[k], float) and d[k] > 0
            for k in ('src_core_mtime', 'src_ai_mtime')))
C.check('B3 ROOT 下扫到的 .py 数在合理区间（只算 src/ + app 层，25–80）',
        25 <= d['n_files'] <= 80, str(d['n_files']))
C.check('B4 newest 里没有 checks/ 或 tempsimu/（不扫非进程代码）',
        all(not it['path'].startswith(('checks/', 'tempsimu/'))
            for it in d['newest']), str([it['path'] for it in d['newest']]))

# ---------------------------------------------------------------
# C. 接线：run_python 返回里带 stale_code_warning
# ---------------------------------------------------------------
print('C) 接线：codeact.run_code 的返回')
from src.ai import codeact                                       # noqa: E402


class _S:
    def get_state(self, scope=None):
        return {}

    def summary(self):
        return {}

    def set_param(self, *a, **k):
        return {}

    def add_level_group(self, *a, **k):
        return {}

    def remove_level_group(self, *a, **k):
        return {}

    def set_waves(self, *a, **k):
        return {}

    def to_dict(self):
        return {'params': {}}

    def restore_snapshot(self, snap):
        return None


_root = tempfile.mkdtemp(prefix='cf_run_')
try:
    out = codeact.run_code('result = 1', _S(), _root, {'seq': [0]},
                           lambda jid: {'ok': True}, lambda jid: {'done': True},
                           lambda jid, s=False: {'ok': True})
    C.check('C1 run_code 返回里**有** stale_code_warning 键',
            'stale_code_warning' in out, str(sorted(out)))
    # 空 root（没有 src/）→ 无从判断 → None（不误报）
    C.check('C2 空 root（无 src/）→ 不误报（None）',
            out['stale_code_warning'] is None, str(out['stale_code_warning']))
    # 造"陈旧"：root 下放个 src/ai/x.py，mtime 晚于本进程基准
    p = os.path.join(_root, 'src', 'ai')
    os.makedirs(p, exist_ok=True)
    f = os.path.join(p, 'x.py')
    with open(f, 'w', encoding='utf-8') as fh:
        fh.write('x = 1\n')
    CF.mark_loaded()                       # 确保基准已盖（first-wins）
    os.utime(f, (time.time() + 5.0, time.time() + 5.0))
    w = CF.stale_warning(root=_root)
    C.check('C3 造陈旧 → stale_warning 含"重启"', isinstance(w, str)
            and '重启' in w, str(w)[:80])
finally:
    shutil.rmtree(_root, ignore_errors=True)

# ---------------------------------------------------------------
# D. 假阴性回归：mark_loaded 必须 first-wins
# ---------------------------------------------------------------
print('D) mark_loaded first-wins（防"基准被推到编辑之后"）')
base0 = CF.baseline()
CF.mark_loaded(time.time() + 1000.0)                # 想推后 → 必须被忽略
C.check('D1 再次 mark_loaded 不改基准（first-wins）',
        CF.baseline() == base0, '%s vs %s' % (CF.baseline(), base0))
C.check('D2 baseline() 是有限浮点', isinstance(base0, float) and base0 > 0)

ok = C.summary()
sys.exit(0 if ok else 1)
