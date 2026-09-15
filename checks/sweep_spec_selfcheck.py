# -*- coding: utf-8 -*-
"""自检：并行扫描的**规格层与策略层**（纯函数；毫秒级、无 Qt、无 ARC）。

覆盖（每段 1 个不变式包，包内子条件单独累计）：
  A 登记册路径寻址：扁平键 / 列表索引 / 索引后字段 / 多级；三种非法路径**报错**
  B 轴展开：显式列表 / linspace / arange / logspace / {'value': …} / 标量
  C 网格：笛卡尔积点数、**顺序确定**（轴名排序、最后一个轴最快）、
    显式 points 追加、空规格报错
  D apply_delta 不改原快照（深拷贝语义）——扫描里 N 个点共用一份 params 的前提
  E 并行度定档 plan_workers：核数上限（留一核）/ 点数上限 / 显式指定夹取 /
    两体按内存限档（给定 avail_gb 时，与 dim² 标定一致）
  F runspec 纯函数：engine./solve. → kwargs 映射（含 None 过滤、diag、pair 剔
    per_freq_M）、sambe+任意波形改道 hard 而 mixed 不改道、per_freq_M 解析
  G **架构守卫**：`import src.core.runspec` 不得拉起 PyQt5 / param_manager
    （runner 子进程要真正 headless；这条是本轮 src/__init__ 惰性化的回归闸）
  H SOLVE_KWARG 与 GUI 侧 `pm_setup._SOLVE_KWARG` 一致（镜像不许漂）

用法：python checks/sweep_spec_selfcheck.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402
from src.sweep import registry as RG                             # noqa: E402
from src.sweep import spec as SP                                 # noqa: E402
from src.core import runspec as RS                               # noqa: E402

C = Checker('sweep_spec')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _raises(fn):
    """fn 抛异常（任意类型）→ True。"""
    try:
        fn()
        return False
    except Exception:                              # noqa: BLE001
        return True


def _Arb():
    """一个真的 ArbitraryWave（改道判据按 isinstance 判，不能用替身）。"""
    from src.core import waveforms as _wf
    return _wf.ArbitraryWave('arb', [0.0, 1.0, 2.0], [0.0, 1.0, 0.0])

PARAMS = {
    'groups.list': [{'n': 60, 'l': 1, 'j': 1.5, 'color': '#f00'}],
    'field.B_gauss': 21.7,
    'field.lights': [{'frequency_hz': 295.7e12, 'intensity': 1.0,
                      'polarization': 'π', 'enabled': True}],
    'dynamic.t_max': 150.0, 'dynamic.npts': 1501,
    'dynamic.method': 'mixed', 'dynamic.mode': 0,
    'dynamic.waves_single': [{'name': 'w', 'freq_mhz': 7380.0,
                              'intensity': 5.0, 'pol': '椭圆偏',
                              'elliptical': {'alpha_deg': 0.0,
                                             'theta_deg': 0.0,
                                             'eps_deg': 0.0},
                              'offset_ns': 0.0,
                              'env': ['gauss', 50.0, 17.0, 1.0],
                              'drive': None}],
    'engine.PHASE_TOL': 1e-4, 'engine.R_CRIT': 0.0135,
    'solve.MAX_STEP': None, 'solve.COST_BUDGET': None,
    'solve.PER_FREQ_M': [1], 'solve.FROZEN_H_NS': 0.0,
}

# ---------------------------------------------------------------
print('A) 登记册路径寻址')
_bad = []
for p in ('a[', 'a[x]', 'a[0]x'):
    try:
        SP.parse_path(p)
        _bad.append(p)
    except ValueError:
        pass
_ok_get = SP.get_in(PARAMS, 'field.lights[0].intensity')
SP.set_in(PARAMS, 'field.lights[0].intensity', 9.0)
_after, PARAMS['field.lights'][0]['intensity'] = (
    PARAMS['field.lights'][0]['intensity'], 1.0)
_missing_key = False
try:
    SP.get_in(PARAMS, 'dynamic.nope')
except KeyError:
    _missing_key = True
_idx_oob = False
try:
    SP.get_in(PARAMS, 'field.lights[7].intensity')
except KeyError:
    _idx_oob = True
C.group('A 路径寻址（parse/get/set + 非法报错）', [
    ('扁平键 parse → 无索引段', SP.parse_path('dynamic.t_max') ==
     ('dynamic.t_max', [])),
    ('索引后字段 parse', SP.parse_path('field.lights[0].intensity') ==
     ('field.lights', [0, 'intensity'])),
    ('取列表内字段', _ok_get == 1.0),
    ('写列表内字段生效并复原', _after == 9.0 and
     PARAMS['field.lights'][0]['intensity'] == 1.0),
    ('未知登记键 → KeyError', _missing_key),
    ('索引越界 → KeyError', _idx_oob),
    ('三种非法路径 → ValueError', not _bad),
])

# ---------------------------------------------------------------
print('B) 轴展开')
C.group('B 轴值六形态', [
    ('显式列表', SP.expand_axis([1, 2, 3]) == [1, 2, 3]),
    ('linspace 含端点', SP.expand_axis({'linspace': [0, 10, 5]}) ==
     [0.0, 2.5, 5.0, 7.5, 10.0]),
    ('arange 不含端点', SP.expand_axis({'arange': [0, 10, 5]}) ==
     [0.0, 5.0, 10.0]),
    ('logspace', SP.expand_axis({'logspace': [1, 100, 3]}) ==
     [1.0, 10.0, 100.0]),
    ("{'value': …} 整个列表当一个取值",
     SP.expand_axis({'value': [1, 2]}) == [[1, 2]]),
    ('标量 = 单点', SP.expand_axis(7) == [7]),
    ('无法识别的 dict → 报错',
     _raises(lambda: SP.expand_axis({'foo': 1}))),
])

# ---------------------------------------------------------------
print('C) 网格（笛卡尔积 + 顺序确定）')
pts = SP.build_points({'b': [1, 2], 'a': [10, 20]})
_again = SP.build_points({'a': [10, 20], 'b': [1, 2]})
_mixed = SP.build_points({'a': [1, 2]}, [{'c': 3}, {'c': 4}])
C.group('C 网格展开', [
    ('2×2 = 4 点', len(pts) == 4),
    ('轴名排序 → 顺序与书写顺序无关', pts == _again),
    ('最后一个轴变化最快',
     [p['b'] for p in pts] == [1, 2, 1, 2] and
     [p['a'] for p in pts] == [10, 10, 20, 20]),
    ('显式 points 追加在后', len(_mixed) == 4 and _mixed[2] == {'c': 3}),
    ('空规格 → 报错', _raises(lambda: SP.build_points({}, []))),
    ('describe_spec 回执点数', SP.describe_spec(
        {'axes': {'a': [1, 2, 3]}})['n_points'] == 3),
])

# ---------------------------------------------------------------
print('D) apply_delta 不改原快照')
before = SP.get_in(PARAMS, 'dynamic.t_max')
p2 = SP.apply_delta(PARAMS, {'dynamic.t_max': 999.0,
                             'field.lights[0].intensity': 42.0})
C.group('D 深拷贝语义（N 点共用一份 params 的前提）', [
    ('原快照标量未变', SP.get_in(PARAMS, 'dynamic.t_max') == before),
    ('原快照嵌套字段未变',
     SP.get_in(PARAMS, 'field.lights[0].intensity') == 1.0),
    ('派生快照已改', SP.get_in(p2, 'dynamic.t_max') == 999.0 and
     SP.get_in(p2, 'field.lights[0].intensity') == 42.0),
    ('两次派生互不影响',
     SP.get_in(SP.apply_delta(PARAMS, {'dynamic.t_max': 1.0}),
               'dynamic.t_max') == 1.0),
])

# ---------------------------------------------------------------
print('E) 并行度定档')
p_cpu = RG.plan_workers(basis_n=20, two_body=False, n_points=100, cpu=16)
p_pts = RG.plan_workers(basis_n=20, two_body=False, n_points=3, cpu=16)
p_req = RG.plan_workers(basis_n=20, two_body=False, n_points=100,
                        requested=4, cpu=16)
p_req_big = RG.plan_workers(basis_n=20, two_body=False, n_points=100,
                            requested=9999, cpu=16)
p_mem = RG.plan_workers(basis_n=68, two_body=True, n_points=100, cpu=16,
                        avail_gb=8.0)
_mem_expect = None
try:
    from src.core.cost import estimate_pair_peak_gb as _g
    _mem_expect = max(1, int((8.0 * 0.4) // _g(68 ** 2)))
except Exception:                                  # noqa: BLE001
    pass
C.group('E plan_workers（内存+核数，含 why 说明）', [
    ('核数上限 = cpu-1', p_cpu['workers'] == 15 and p_cpu['lim_cpu'] == 15),
    ('点数少于核数 → 取点数', p_pts['workers'] == 3),
    ('显式指定优先', p_req['workers'] == 4),
    ('显式指定做合理性夹取', p_req_big['workers'] == 64),
    ('两体按 dim² 内存限档', p_mem['workers'] == _mem_expect),
    ('两体档位确实低于核数上限（内存是紧的那一头）',
     p_mem['workers'] < p_mem['lim_cpu']),
    ('why 是人话（非空且含"自动"）', bool(p_cpu['why']) and
     '自动' in p_cpu['why']),
    ('单位峰值可回读', p_mem['per_worker_gb'] and
     p_mem['per_worker_gb'] > 0),
])

# ---------------------------------------------------------------
print('F) runspec 纯函数（kwargs 映射 / 改道规则）')
ek = RS.god_kwargs_from_params(PARAMS, diag=False)
ekd = RS.god_kwargs_from_params(PARAMS, diag=True)
ekp = RS.god_kwargs_from_params(PARAMS, pair=True)
C.group('F runspec 纯函数', [
    ('engine.* → param_overrides',
     ek.get('param_overrides', {}).get('PHASE_TOL') == 1e-4),
    ('solve.MAX_STEP（None）被过滤', 'max_step' not in ek),
    ('solve.PER_FREQ_M → per_freq_M', ek.get('per_freq_M') == [1]),
    ('solve.FROZEN_H_NS=0.0 保留', ek.get('frozen_h_ns') == 0.0),
    ('diag=True 时带 diag', ekd.get('diag') is True and 'diag' not in ek),
    ('pair=True 剔除 per_freq_M', 'per_freq_M' not in ekp),
    ('sambe + 任意波形 → 改道 hard 且有说明',
     RS.effective_method('sambe', [[_Arb()]])[0] == 'hard' and
     RS.effective_method('sambe', [[_Arb()]])[1]),
    ('mixed + 任意波形 → 不改道',
     RS.effective_method('mixed', [[_Arb()]])[0] == 'mixed'),
    ('无任意波形 → 原样',
     RS.effective_method('sambe', [[]])[0] == 'sambe'),
    ('parse_per_freq_m：[1] / 2 / "3" / 坏值',
     RS.parse_per_freq_m([1]) == [1] and RS.parse_per_freq_m(2) == 2 and
     RS.parse_per_freq_m('3') == 3 and RS.parse_per_freq_m('xx') is None),
])

# ---------------------------------------------------------------
print('G) 架构守卫：core 导入不得拉起 Qt / param_manager')
code = ("import sys; sys.path.insert(0, r'%s');"
        "import src.core.runspec;"
        "print(int('PyQt5' in sys.modules),"
        "int('param_manager' in sys.modules))" % ROOT)
try:
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=120)
    toks = (out.stdout or '').strip().split()
    ok_parse = (out.returncode == 0 and len(toks) == 2)
    qt_loaded = (toks[0] == '1') if ok_parse else None
    pm_loaded = (toks[1] == '1') if ok_parse else None
    detail = (out.stderr or '')[-300:]
except Exception as e:                             # noqa: BLE001
    ok_parse, qt_loaded, pm_loaded, detail = False, None, None, str(e)
C.group('G src.core 导入的 Qt 洁净度（runner 真 headless 的根据）', [
    ('子进程跑通并回两个标志', ok_parse),
    ('import src.core.runspec 未加载 PyQt5', qt_loaded is False),
    ('import src.core.runspec 未加载 param_manager', pm_loaded is False),
], detail)

# ---------------------------------------------------------------
print('H) SOLVE_KWARG 镜像一致性（GUI 侧定义点 vs 无 Qt 侧）')
# ⚠ 2026-09-15 改（拷贝环境实测）：原来 `import src.pm_setup` 需要兄弟目录
# 'Param manager'（pm_setup 逐级向上找 6 层）⇒ **项目一旦离开原始父目录**
# （拷贝/分发/换机器）就导入失败 ⇒ 这条"镜像不许漂"的守卫**永久 SKIP**，
# 即零覆盖（项目最忌的假绿）。改为 **AST 解析源码**取常量：不需要导入、
# 不需要 Qt/param_manager ⇒ 任何环境下守卫都活着；导入仅作兜底。
import ast as _ast                                               # noqa: E402


def _solve_kwarg_from_source(path):
    """从源码取 `_SOLVE_KWARG`（**不导入 pm_setup**）。

    实测该常量不是纯字面量（`literal_eval` 报 malformed node）⇒ 用
    `ast.get_source_segment` 取右值源码片段，再在**受限命名空间**里求值：
    `__builtins__` 只放行 `dict`，因此只能构造字典/字面量，无法执行任意代码。
    """
    _txt = open(path, encoding='utf-8').read()
    _tree = _ast.parse(_txt)
    _ns = {'__builtins__': {'dict': dict}}
    # 两遍：`_SOLVE_KWARG` 的右值会引用别的模块级常量（实测 `_SOLVE_SPECS`），
    # 故先把模块级可受限求值的常量都收进命名空间，再取目标。
    for _pass in range(2):
        for _n in _tree.body:
            if not isinstance(_n, _ast.Assign):
                continue
            for _t in _n.targets:
                if not isinstance(_t, _ast.Name):
                    continue
                _rhs = _ast.get_source_segment(_txt, _n.value)
                if _rhs is None:
                    continue
                try:
                    _ns[_t.id] = eval(_rhs, dict(_ns))        # noqa: S307
                except Exception:                             # noqa: BLE001
                    pass
    if '_SOLVE_KWARG' not in _ns:
        raise LookupError('源码里取不到 _SOLVE_KWARG')
    return dict(_ns['_SOLVE_KWARG'])


gui_kw, gui_how, _err = None, '', ''
try:
    gui_kw = _solve_kwarg_from_source(os.path.join(ROOT, 'src', 'pm_setup.py'))
    gui_how = '源码 AST（无需导入）'
except Exception as _e1:                           # noqa: BLE001
    try:
        import src.pm_setup as pm
        gui_kw = dict(pm._SOLVE_KWARG)
        gui_how = '导入 pm_setup（AST 兜底失败：%s）' % str(_e1)[:40]
    except Exception as _e2:                       # noqa: BLE001
        _err = '%s / %s' % (str(_e1)[:40], str(_e2)[:60])
if gui_kw is None:
    C.skip('H SOLVE_KWARG 镜像一致性',
           '源码与导入都取不到（%s）' % _err)
else:
    same = dict(RS.SOLVE_KWARG) == dict(gui_kw)
    C.group('H solve.* → kwarg 映射不许漂（两处必须一致）', [
        ('与 pm_setup._SOLVE_KWARG 完全相同（取自%s）' % gui_how, same),
    ], 'gui=%s runner=%s' % (dict(gui_kw), dict(RS.SOLVE_KWARG)))

# ---------------------------------------------------------------
print('I) 并发读写 status.json 不抛（Windows WinError 5 回归）')
# 由来（2026-09-15 聚合回归抓到）：runner 每秒 `os.replace` status.json，而
# GUI/MCP 每秒读同一份 → 撞上时 `os.replace` 抛 PermissionError(WinError 5)，
# **整轮扫描被标 failed**（4 点里已完成 3 点）。修法 = 重试 + 退路直写；
# 这条用真并发把"读者一直读、写者一直写"钉住。
import json as _json                                             # noqa: E402
import tempfile                                                  # noqa: E402
import threading                                                 # noqa: E402
from src.sweep import runner as RN                               # noqa: E402

_tmpd = tempfile.mkdtemp(prefix='sweep_rw_')
_p = os.path.join(_tmpd, 'status.json')
_err = []
_stop = threading.Event()
_reads = [0, 0]                     # [成功读, 读不到（半截/锁住）]


def _reader():
    while not _stop.is_set():
        try:
            with open(_p, 'r', encoding='utf-8') as f:
                _json.load(f)
            _reads[0] += 1
        except FileNotFoundError:
            _reads[1] += 1
        except Exception:                            # noqa: BLE001 - 半截内容
            _reads[1] += 1


_th = threading.Thread(target=_reader, daemon=True)
RN._write_json(_p, {'n': -1})                        # 先建出来
_th.start()
try:
    for i in range(150):
        try:
            RN._write_json(_p, {'n': i, 'pad': 'x' * 400})
        except Exception as e:                       # noqa: BLE001
            _err.append('%s: %s' % (type(e).__name__, e))
finally:
    _stop.set()
    _th.join(timeout=5)
try:
    with open(_p, 'r', encoding='utf-8') as f:
        _final = _json.load(f)
except Exception as e:                               # noqa: BLE001
    _final = {'__err__': str(e)}
import shutil as _shutil                             # noqa: E402
_shutil.rmtree(_tmpd, ignore_errors=True)
C.group('I status.json 并发读写：写者不抛 + 内容始终可解析', [
    ('150 次写全部无异常（WinError 5 已修）%s' % (_err[:1] or ''), not _err),
    ('读者读到的永远是**完整 JSON**（不是半截）', _reads[0] > 0),
    ('末次内容可解析且是最后写入的值', _final.get('n') == 149),
    ('退路直写也在（函数签名带 retries/backoff）',
     'retries' in RN._write_json.__code__.co_varnames),
])

sys.exit(0 if C.summary() else 1)
