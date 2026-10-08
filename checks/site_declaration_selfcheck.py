# -*- coding: utf-8 -*-
"""站点声明自检（统一启动器 · 步骤 1）——**结构 + 单一事实 + CLI 契约**（不起真进程）。

四条（每条都指向设计文档 `docs/v2/统一启动器-设计.md` 的一节）：

- **A** 声明能被框架按约定加载：`mecha.launch.load_sites()`（`MECHA_SITE_ROOTS` 指本仓根）
  ⇒ 拿到 `Site`、`services` 里有 `EL`、`validate()` 不抛（框架在加载时已调一次，这里显式再来）；
- **B** `ready` 的**跨语言单一事实**：声明的 `port_file` 必须与 Python 侧
  （`src/v2_bind.py` 的默认发现文件）**和** dsh 插件侧（`dsh/src/el-params.ts` 的 `mcpPortFile`）
  是**同一个名字**——三处任一漂了，"面板连不上/服务判不就绪"就会在**两个语言**之间对不上。
  ⚠ 同时钉住 `ok_codes=()`：**实测** `GET /mcp` 回 406（该端口上无 2xx 路由）⇒ 不许改回默认；
- **C** ⭐ **状态根环境变量已钉死**（§5.6）：`RE0_V2_DATA_ROOT` 必须等于声明的 `data_root`
  （否则换腿后看到另一份 journal，且**无任何报错**）；`EL_NO_AI_SERVER` 必须**存在且为假值**
  （真值会让 MCP 服务端**根本不起**）；`PYTHONIOENCODING` 为 `utf-8`（日志重定向下防 cp936 崩）；
- **D** **CLI 契约已改名**：`app_entry.py` 认 `--mcp`、不再认 `--authority`、无参时指向
  `python -m mecha`（⚠ **不是** `mecha.launch`：那是模块不是包，会静默什么都不做）；
  `launcher.py`（步骤 4 才删）在共存期 spawn 的也是 `--mcp`。

真进程冒烟在另一条判据（`checks/mcp_entry_smoke_selfcheck.py`）：那条要起真服务、秒级起不来，
**故意不进 `--core/--fast`**（档位纪律：那两档是秒级）。
"""
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
    ('⭐ `manual` 已按三层架构删掉（它曾是「要 spawn 的进程」，现在是进程内窗口）',
     _svc is not None and not hasattr(_svc, 'manual')),
    ('has_gui=True（EL 有 GUI 窗口 ⇒ 面板显示「切到手动」）',
     _svc is not None and getattr(_svc, 'has_gui', None) is True),
    ('⭐ initial_mode=AI（启动者声明：从 mecha 启动永远是 AI 组）',
     # ⚠ **判据自己踩过这个坑**：`Mode` 是 `str` 枚举 ⇒ `str(Mode.AI)` 给的是
    # `'Mode.AI'` 而**不是** `'ai'` ⇒ 必须比 `.value`（框架里同款坑：`SvcState.DOWN`）。
    _svc is not None
    and getattr(getattr(_svc, 'initial_mode', None), 'value', None) == 'ai'),
    ('argv 起的是 --mcp 形态，且 cwd 是本仓根',
     _svc is not None and '--mcp' in _svc.argv
     and Path(_svc.cwd).resolve() == ROOT),
    ('data_root 是本仓根（写租约根 = .mecha-v2 的上一层）',
     _svc is not None and Path(_svc.data_root).resolve() == ROOT),
])

# ---------------------------------------------------------------- B. ready 的单一事实
_pf = _svc.ready.port_file if (_svc and _svc.ready) else None
_ok_codes = _svc.ready.ok_codes if (_svc and _svc.ready) else 'N/A'
_v2 = (ROOT / 'src' / 'v2_bind.py').read_text(encoding='utf-8')
_ts = (ROOT / 'dsh' / 'src' / 'el-params.ts').read_text(encoding='utf-8')
C.group('B ready：端口文件名是**跨语言单一事实**，且 ok_codes 与实测一致', [
    ('声明里的 port_file = %r' % _pf, _pf == '.mcp-port'),
    ('Python 侧（src/v2_bind.py）用的是同一个名字',
     _pf is not None and ('"%s"' % _pf) in _v2),
    ('dsh 插件侧（dsh/src/el-params.ts 的 mcpPortFile）也是同一个名字',
     _pf is not None and ("mcpPortFile: '%s'" % _pf) in _ts),
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

# ---------------------------------------------------------------- D. CLI 契约
_app = (ROOT / 'app_entry.py').read_text(encoding='utf-8')
_launch = (ROOT / 'launcher.py').read_text(encoding='utf-8')
C.group('D CLI 契约：形态改名成 --mcp，无参指向 python -m mecha', [
    ("app_entry.py 认 '--mcp'", "'--mcp' in _app or '\"--mcp\"' in _app"),
    ("app_entry.py 不再认 '--authority'（契约层清干净）", '--authority' not in _app),
    ('无参时指向 python -m mecha --all，且**不再转发给看门人**（import launcher 已去；'
     '⚠ 注释里那句“不是 mecha.launch”是刻意的提醒，不算违规）',
     'python -m mecha --all' in _app and 'import launcher' not in _app),
    ('launcher.py（步骤 4 才删）共存期 spawn 的也是 --mcp', '"--mcp"' in _launch),
    ('launcher.py 里不再引用 --authority', '--authority' not in _launch),
    ("MCP 服务形态的函数是 run_mcp", 'def run_mcp' in _app),
])

sys.exit(0 if C.summary() else 1)
