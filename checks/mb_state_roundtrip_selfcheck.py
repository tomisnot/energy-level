# -*- coding: utf-8 -*-
"""多体页「状态保存 / 导入」标杆：演示存档可加载、界面真的同步、且真能跑出结果。

为什么单独一条判据：会话是真值源，但**三类状态不在会话里**——
① 登记型控件只在构造时 seed 一次，外部写（恢复存档、AI/MCP `set_config`）不回灌；
② 能级组住在 `LevelGroupStore`；③ 点云住在多体页 `_atoms` 缓存。
实测踩过：只 `load()` 不拉同步 → 后端下拉仍显示 auto（会话其实已是 dmrg）、
原子表为空、且因 `main.groups` 未更新而取不到 g/r（多体页误报“需恰好 2 个能级”）。

**耗时纪律**（用户要求全测压在 3 分钟内）：本判据只做"导入 + 界面同步"（8 份档，
无解）与**一个小尺寸真跑**；大尺寸的跨后端一致性归 `equibench_dmrg_cluster`，
产品路径归 `equibench_product`——不在此重复烧算力。

offscreen；os._exit 绕过 Qt teardown 噪声。
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')
os.environ['RE0_V2_DATA_ROOT'] = tempfile.mkdtemp(prefix='mbstates_')

import numpy as np                                       # noqa: E402
from PyQt5.QtWidgets import QApplication                 # noqa: E402
from checks.checklib import Checker                      # noqa: E402

C = Checker('mb_state_roundtrip')
app = QApplication.instance() or QApplication(sys.argv)

import app_entry                                         # noqa: E402
from src.core import runspec as RS                       # noqa: E402

SD = os.path.join(ROOT, 'save_state')
DEMOS = [
    '多体-平衡态-基态-链8-ED.json',
    '多体-平衡态-基态-2D-3x4-ED.json',
    '多体-平衡态-基态-2D-4x4-Lanczos.json',
    '多体-平衡态-基态-2D-4x4-DMRG.json',
    '多体-平衡态-基态-2D-2x10-DMRG超ED.json',
    '多体-平衡态-有限温KPM-3x3.json',
    '多体-平衡态-态密度DOS-2x6.json',
    '多体-时间演化-阻塞链8.json',
]

missing = [f for f in DEMOS if not os.path.exists(os.path.join(SD, f))]
C.group('演示存档齐备（由 scripts/make_demo_states.py 生成）', [
    ('8 份档都在 save_state/（缺：%s）' % missing, not missing),
])
if missing:
    sys.stdout.flush()
    os._exit(1)

w = app_entry.MainWindow()
br = w.app_bridge
mb = w.manybody

# ---------------- A. 逐份导入 → 界面与会话一致（不求解，只查同步） ----------------
subs_i = []
for name in DEMOS:
    with open(os.path.join(SD, name), encoding='utf-8') as f:
        want_file = json.load(f)
    n_atoms = len(want_file.get('manybody.atoms') or [])
    want = want_file.get('manybody.want') or ''
    backend = want_file.get('manybody.eq_backend')
    betas = want_file.get('manybody.betas')
    tag = name.replace('多体-', '')[:30]
    try:
        br.load(os.path.join(SD, name), reason='判据：导入', actor='gui')
        w._resync_from_session()
        app.processEvents()
    except Exception as e:                              # noqa: BLE001
        subs_i.append(('%s：导入抛错 %s' % (tag, type(e).__name__), False))
        continue
    sess = br.effective()
    lv = mb._levels()
    subs_i.append(('%s → 能级 %d 个、表 %d 行、模式 %s'
                   % (tag, len(lv), mb.table.rowCount(), mb._mode),
                   len(lv) == 2                            # g/r 来自能级组
                   and mb.table.rowCount() == n_atoms      # 点云回显
                   and len(mb._atoms) == n_atoms
                   and mb._mode == ('eq' if want else 'evolve')
                   and sess.get('manybody.want') == want
                   and len(sess.get('groups.list') or []) == 2))
    if want:            # 平衡态档：请求量与后端必须回显成档里的值（不是控件默认值）
        subs_i.append(('%s → 请求量 %s / 后端 %s 回显'
                       % (tag, mb.want_cb.currentText(),
                          mb.backend_cb.currentText()),
                       mb.want_cb.currentText() == want
                       and mb.backend_cb.currentText() == backend))
    if betas:
        got = [float(x) for x in mb.betas_edit.text().replace(',', ' ').split()]
        subs_i.append(('%s → betas 回显 %s' % (tag, got), got == betas))
C.group('A 导入后界面与会话同步（能级/表行数/模式/请求量/后端/betas）', subs_i)

# ---------------- B. g/r 单一真源 = 能级组（档里的旧值要被纠正） ----------------
with open(os.path.join(SD, DEMOS[0]), encoding='utf-8') as f:
    raw = json.load(f)
raw['manybody.atoms'] = [dict(a, g=[1.0, 2.0, 3.0, 4.0], r=[5.0, 6.0, 7.0, 8.0])
                         for a in raw['manybody.atoms']]
tmp = os.path.join(tempfile.mkdtemp(prefix='gr_'), 'gr.json')
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(raw, f, ensure_ascii=False)
br.load(tmp, reason='判据：错 g/r 档', actor='gui')
w._resync_from_session()
app.processEvents()
lv = mb._levels()
stored = br.read('manybody.atoms') or []
subs_g = [('能级组给出恰好 2 个 m_J 能级', len(lv) == 2)]
if len(lv) == 2:
    subs_g.append(('档里错的 g/r 被能级组覆盖（g=%s、r=%s）'
                   % (stored[0]['g'], stored[0]['r']),
                   all(list(a['g']) == list(lv[0]) and list(a['r']) == list(lv[1])
                       for a in stored)))
C.group('B g/r 单一真源 = 能级组（导入档里的旧 g/r 会被纠正）', subs_g)

# ---------------- C. 真跑（只挑小尺寸，控总时长） ----------------
subs_r = []


def _run_one(fname):
    br.load(os.path.join(SD, fname), reason='判据：真跑', actor='gui')
    w._resync_from_session()
    app.processEvents()
    snap = br.effective()
    try:
        spec = RS.build_run(w.cache, snap)
        probs = RS.check_run(spec)
        res = RS.evolve_run(spec)
        return spec, res, probs, None
    except Exception as e:                              # noqa: BLE001
        return None, None, None, '%s: %s' % (type(e).__name__, e)


sp, res, probs, err = _run_one(DEMOS[0])       # 链8 gs / auto→ed（~1 s）
subs_r.append(('gs 链8：build_run→evolve_run 通、auto 走 ed、check_run 无告警（%s）'
               % (err or 'ok'),
               err is None and res['want'] == 'gs' and np.isfinite(res['E0'])
               and res.get('backend') == 'ed' and not probs))
rsum = RS.result_summary(sp, res) if err is None else {}
subs_r.append(('gs 链8：摘要 ok 且带 E₀ 与 8 个 ⟨n_i⟩',
               rsum.get('ok') is True and rsum.get('E0') is not None
               and len(rsum.get('n_atom') or []) == 8))
_na = np.asarray(rsum.get('n_atom') or [0.0], float)
subs_r.append(('演示参数非平庸：⟨n_i⟩∈[%.3f, %.3f] 既非全 0 也非全 1'
               % (_na.min(), _na.max()),
               _na.size and float(_na.max()) < 0.95 and float(_na.min()) > 0.02))
sp2, res2, probs2, err2 = _run_one('多体-时间演化-阻塞链8.json')   # 含时（2^8 小）
subs_r.append(('含时链8：走演化路径（有 t/pops、无 equilibrium 标记、按档取 npts）'
               '（%s）' % (err2 or 'ok'),
               err2 is None and not sp2.get('equilibrium') and 'pops' in res2
               and len(res2['t']) == len(sp2['t'])))
C.group('C 导入后真能跑（小尺寸各一遍：基态 + 含时）', subs_r)

# ---------------- D. 图上中文必须有字体（ARC 的回归守卫） ----------------
# 真因：`arc` 在 import 时把 rcParams['font.family'] 改成 'serif'（它自带的绘图风格），
# 而 app_entry 早于 arc 导入就设了中文字体 ⇒ 被事后覆盖 ⇒ 图上中文全变方框。
# 现 app_entry 先 import arc 再设字体；本组钉住“设定时机”与“不缺字形”。
import matplotlib                                    # noqa: E402
import warnings                                      # noqa: E402

fam = list(np.atleast_1d(matplotlib.rcParams['font.family']).astype(str))
sans = list(np.atleast_1d(matplotlib.rcParams['font.sans-serif']).astype(str))
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter('always')
    mb._draw_eq_result({'want': 'gs', 'backend': 'ed', 'n_atoms': 4, 'dim': 16,
                        'E0': -1.23, 'gap': 0.4,
                        'n_atom': np.linspace(0.1, 0.4, 4)})
    mb.eq_canvas.draw()
    miss = [str(x.message) for x in ws if 'missing from' in str(x.message)]
C.group('D 图上中文有字体（防 ARC 把 font.family 改回 serif）', [
    ("font.family == sans-serif（实=%s）" % fam,
     fam == ['sans-serif']),
    ('sans-serif 首选是中文字体（实=%s）' % sans[:1],
     sans and sans[0] in ('Microsoft YaHei', 'SimHei')),
    ('画一次平衡态结果图：零缺字形告警（实 %d 条：%s）'
     % (len(miss), miss[:1]), not miss),
    ('结果图真的画出了柱子（不是异常占位）',
     len(mb.eq_ax_main.patches) == 4),
])

sys.stdout.flush()
os._exit(0 if C.summary() else 1)
