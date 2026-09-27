# -*- coding: utf-8 -*-
"""**启动初值被拒必须留痕**（D5 同源：拒绝也是一件事）。

## 为什么有这条

框架 2026-09-26（A2）给 `Gate.seed` 补上了**值校验**（seed 仍不过写权门）⇒ 非法初值会被拦。
而 EL 的 GUI 绑定层当年两处种子点都是 `except Exception: pass`：

```
src/ui/_mecha_bind.py  _seed() / push_all()   ← 静默吞（改动前）
```

⇒ 拒绝被吞掉后，**"种了但被拒"与"从没种过"完全无法区分**（快照不动、史不追加），
**把补校验的意义整个抵消**。本判据守"**留得下痕**"：

1. **日志**：打出校验器给的原话（kind + 键名 + 去向）；
2. **史**：经 `Gate.record` 记一条**审计事件**（只记史、不改快照）——键用 `seed.rejected`，
   值里带被拒的键名、kind、原话、hint。

## 三组断言

- **A 被拒留痕**（真校验器、真键 `dynamic.pair`——v2 里不在可写清单）：
  快照**没有**该键；史里**有** `seed.rejected` 事件且值指向该键 + `kind=unknown_key`；
  日志里出现键名与 kind。
- **B 成功不误报**（对照：在册键 `dynamic.decays`）：值**真落快照**，且**不新增**
  `seed.rejected`（R11/R8：观测量必须能变，否则"永远报被拒"也会过）。
- **R8 非退化自证**：打印事件数与拒绝数；断言两者都 > 0。

offscreen（QT_QPA_PLATFORM=offscreen）；临时 root，不污染 `.mecha`。
用法：python checks\\seed_rejection_visible_selfcheck.py     退出码 0 全过 / 1 有问题。
"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from PyQt5.QtWidgets import QApplication                      # noqa: E402
from checks.checklib import Checker                           # noqa: E402
from mecha_v2.bridge import V2Bridge                          # noqa: E402
from src.ui import _mecha_bind as MB                          # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)       # offscreen
C = Checker('seed_reject')
EL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tmp = tempfile.mkdtemp(prefix='seedreject_')
br = V2Bridge.for_shared(el_root=EL_ROOT, data_root=tmp, mode='human')
binder = MB.Binder(br)
REJECTED_KEY = 'dynamic.pair'          # v2 不在可写清单（旧字典键；v2 注释亦已说明）
OK_KEY = 'dynamic.decays'              # 在册（判据 gui_bind_selfcheck 也种它）


def _events():
    return list(br._history.events())


def _rejections():
    return [e for e in _events() if e.key == 'seed.rejected']


# ---------------- A 被拒：必须留痕 ----------------
before = len(_events())
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    binder.register_external(REJECTED_KEY, read_fn=lambda: {'R_um': 5.0},
                             write_fn=lambda v: None)
logged = buf.getvalue()
rej = _rejections()
info = (rej[-1].value if rej else {}) or {}
C.group('A 初值被拒要留痕：快照不动 + 史里有 seed.rejected + 日志有原话', [
    ('快照里没有被拒的键（br.read → %r）' % br.read(REJECTED_KEY),
     br.read(REJECTED_KEY) is None),
    ('史里新增 seed.rejected（%d 条；值=%s）'
     % (len(rej), {k: info.get(k) for k in ('key', 'kind')}),
     bool(rej) and info.get('key') == REJECTED_KEY),
    ('记的 kind 是校验器原话 unknown_key（实得 %r）' % info.get('kind'),
     info.get('kind') == 'unknown_key'),
    ('日志打出了键名与 kind', REJECTED_KEY in logged and 'unknown_key' in logged),
    ('事件数确实增加（%d → %d）' % (before, len(_events())), len(_events()) > before),
])

# ---------------- B 成功：不误报（观测量能变） ----------------
before2 = len(_rejections())
n_ev0 = len(_events())
binder.register_external(OK_KEY, read_fn=lambda: [{'gamma_mhz': 1.0}],
                         write_fn=lambda v: None)
C.group('B 成功路径不误报：在册键真落快照，且不新增 seed.rejected', [
    ('在册键初值真落快照（%r）' % (br.read(OK_KEY),),
     br.read(OK_KEY) == [{'gamma_mhz': 1.0}]),
    ('没有多出 seed.rejected（%d → %d）' % (before2, len(_rejections())),
     len(_rejections()) == before2),
    ('史确实长了（%d → %d）' % (n_ev0, len(_events())), len(_events()) > n_ev0),
])

# ---------------- R8 自证 ----------------
print('[seed_reject] 非退化自证：事件 %d 条、拒绝 %d 条（被拒键 %r / 通过键 %r）'
      % (len(_events()), len(_rejections()), REJECTED_KEY, OK_KEY))
C.group('R8 非退化自证：两个观测量都非零（否则"恒报/恒不报"也会过）', [
    ('拒绝事件 %d > 0' % len(_rejections()), len(_rejections()) > 0),
    ('通过路径真落了史（%d > 0）' % len(_events()), len(_events()) > 0),
])

sys.exit(0 if C.summary() else 1)
