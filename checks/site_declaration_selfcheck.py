# -*- coding: utf-8 -*-
"""站点加载、就绪配置与状态根优先级。实际启动由 mcp_entry_smoke 验证。
不扫 CLI 或 TypeScript 的固定源码拼写。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402

C = Checker('site_declaration')
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- A. 框架能加载
_err = None
_site = None
_svc = None
try:
    from mecha.launch import load_sites
    _old = os.environ.get("MECHA_SITE_ROOTS")
    os.environ["MECHA_SITE_ROOTS"] = str(ROOT)
    try:
        _sites = load_sites()
    finally:
        if _old is None:
            os.environ.pop("MECHA_SITE_ROOTS", None)
        else:
            os.environ["MECHA_SITE_ROOTS"] = _old
    _site = next((s for s in _sites if any(v.id == "EL" for v in s.services)), None)
    _svc = next((v for v in (_site.services if _site else ()) if v.id == "EL"), None)
except Exception as e:                                        # noqa: BLE001
    _err = '%s: %s' % (type(e).__name__, e)

C.group('A 框架能按约定加载本仓声明（load_sites + validate）', [
    ('找到 id=EL 的站点（%s）' % (_err or 'ok'), _svc is not None),
    ('validate() 通过（id 形状 / dsh 插件产物都在）',
     _site is not None and _err is None),
    ('has_gui=True（EL 有 GUI 窗口 ⇒ 面板显示「切到手动」）',
     _svc is not None and getattr(_svc, 'has_gui', None) is True),
    ('⭐ initial_mode=AI（启动者声明：从 mecha 启动永远是 AI 组）',
     # ⚠ **判据自己踩过这个坑**：`Mode` 是 `str` 枚举 ⇒ `str(Mode.AI)` 给的是
    # `'Mode.AI'` 而**不是** `'ai'` ⇒ 必须比 `.value`（框架里同款坑：`SvcState.DOWN`）。
    _svc is not None
    and getattr(getattr(_svc, 'initial_mode', None), 'value', None) == 'ai'),
    ('cwd 是本仓根，且 argv 起的是 mecha 服务形态（形参见 `service_argv()` 那**一处**）',
     #: ⚠ **2026-10-10：这里原先把旗标拼写钉死（`'--mcp' in argv` / `'--with-gui' not in argv`），
     #: 已按用户裁决删掉** —— *"三家 CLI 按词汇表改名，但是相关判据删掉，不要什么都加判据"*。
     #: 旗标拼写属于**接入指南的词汇表约定**，不是产品行为；钉它只会让每次改名都动判据。
     #: 形态的唯一来源是 `mecha_site.service_argv()` 那一处（读代码比读判据快）。
     _svc is not None and len(_svc.argv) > 1
     and Path(_svc.cwd).resolve() == ROOT),
    ('data_root 是本仓根（写租约根 = .mecha-v2 的上一层）',
     _svc is not None and Path(_svc.data_root).resolve() == ROOT),
])

# ---------------------------------------------------------------- B. ready 的单一事实
_pf = _svc.ready.port_file if (_svc and _svc.ready) else None
_ok_codes = _svc.ready.ok_codes if (_svc and _svc.ready) else 'N/A'
C.group('B ready：端口文件名是**跨语言单一事实**（软件 ↔ 声明），服务探活配置', [
    ('声明里的 port_file = %r' % _pf, _pf == '.mcp-port'),
    ('ok_codes=()（实测 GET /mcp 回 406，该端口无 2xx 路由 ⇒ 不放宽则永远不就绪）',
     isinstance(_ok_codes, tuple) and len(_ok_codes) == 0),
    ('probe 打的是 /mcp（不是根路径：GET / 回 404）',
     _svc is not None and _svc.ready.probe is not None
     and _svc.ready.probe.endswith('/mcp')),
])

# ---------------------------------------------------------------- C. 状态根钉死
_env = dict(_svc.env) if _svc else {}
C.group('C ⭐ 影响“看到哪份状态”的环境变量已由声明显式钉死（§5.6）', [
    ('RE0_V2_DATA_ROOT 已钉死，且 == data_root（否则换腿后看到另一份 journal，无报错）',
     _env.get('RE0_V2_DATA_ROOT') is not None
     and Path(_env['RE0_V2_DATA_ROOT']).resolve() == ROOT),
    ('EL_NO_AI_SERVER 存在且为**假值**（真值 ⇒ v2_bind 关掉后台服务 ⇒ 服务端根本不起）',
     'EL_NO_AI_SERVER' in _env and not _env['EL_NO_AI_SERVER']),
    ('PYTHONIOENCODING=utf-8（stdout 被重定向 ⇒ 防 cp936 崩）',
     _env.get('PYTHONIOENCODING') == 'utf-8'),
])

# ---------------------------------------------------------------- E. 数据根同源（§5.1 / §5.6）
# ⭐ 两个变量：`MECHA_SOFTWARE_ROOT`（**框架声明**的数据根）与 `RE0_V2_DATA_ROOT`
# （EL **内部**管线变量：判据/测试用它指临时目录）。若各按各的解析 ⇒ **看到两份状态且无报错**。
# 定案顺序写在 `resolve_data_root` 的 docstring 里：**显式 > 框架声明 > EL 内部 > el_root**；
# 两者都设且不一致 ⇒ 采信框架的，但**响亮说出来**（不静默）。
import contextlib                                                  # noqa: E402
import io as _io                                                   # noqa: E402
import os as _os                                                   # noqa: E402
sys.path.insert(0, str(ROOT))
from src.v2_bind import resolve_data_root                          # noqa: E402

_KEYS = ('MECHA_SOFTWARE_ROOT', 'RE0_V2_DATA_ROOT')


def _probe(env: dict, explicit=None):
    """在受控环境里问一次解析结果 → (结果, 有没有出声)。"""
    keep = {k: _os.environ.get(k) for k in _KEYS}
    buf = _io.StringIO()
    try:
        for k in _KEYS:
            _os.environ.pop(k, None)
        for k, v in env.items():
            if v is not None:
                _os.environ[k] = v
        with contextlib.redirect_stdout(buf):
            got = resolve_data_root(explicit, '/ELROOT')
        return got, buf.getvalue()
    finally:
        for k, v in keep.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v


_g_mecha, _w_mecha = _probe({'MECHA_SOFTWARE_ROOT': '/FWROOT'})
_g_re0, _w_re0 = _probe({'RE0_V2_DATA_ROOT': '/TMPROOT'})
_g_same, _w_same = _probe({'MECHA_SOFTWARE_ROOT': '/X', 'RE0_V2_DATA_ROOT': '/X'})
_g_diff, _w_diff = _probe({'MECHA_SOFTWARE_ROOT': '/FWROOT', 'RE0_V2_DATA_ROOT': '/TMPROOT'})
_g_none, _w_none = _probe({})
_g_exp, _w_exp = _probe({'MECHA_SOFTWARE_ROOT': '/FWROOT', 'RE0_V2_DATA_ROOT': '/TMPROOT'},
                        explicit='/EXPLICIT')

C.group('E ⭐ 数据根同源：框架声明 > EL 内部 > el_root；不一致**出声**（§5.1/§5.6）', [
    ('只设框架声明 ⇒ 用它（%s）' % _g_mecha, _g_mecha == '/FWROOT'),
    ('只设 EL 内部（判据/测试形态）⇒ 用它（%s）' % _g_re0, _g_re0 == '/TMPROOT'),
    ('两者都设且**相同** ⇒ 用它、且**不出声**（%s）' % _g_same,
     _g_same == '/X' and not _w_same.strip()),
    ('⭐ 两者都设且**不同** ⇒ 采信**框架声明**（%s）' % _g_diff, _g_diff == '/FWROOT'),
    ('⭐ 且**响亮说出来**（两个值都点名）（%s）' % ('有' if _w_diff.strip() else '无'),
     'MECHA_SOFTWARE_ROOT' in _w_diff and 'RE0_V2_DATA_ROOT' in _w_diff
     and _g_diff in _w_diff),
    ('显式参数优先于两个环境变量（%s）' % _g_exp, _g_exp == '/EXPLICIT'),
    ('都没设 ⇒ 回落 el_root（%s）' % _g_none, _g_none == '/ELROOT'),
])

sys.exit(0 if C.summary() else 1)
