# -*- coding: utf-8 -*-
"""站点声明自检（统一启动器 · 步骤 1）——**结构 + 单一事实 + CLI 契约**（不起真进程）。

四条（每条都指向声明契约的一节）：

- **A** 声明能被框架按约定加载：`mecha.launch.load_sites()`（`MECHA_SITE_ROOTS` 指本仓根）
  ⇒ 拿到 `Site`、`services` 里有 `EL`、`validate()` 不抛（框架在加载时已调一次，这里显式再来）；
- **B** `ready` 的**跨语言单一事实**：声明的 `port_file` 必须与软件侧
  （`src/v2_bind.py` 的默认发现文件）是**同一个名字**——两处任一漂了，"服务判不就绪"就会
  在**两个语言**之间对不上。
  ⭐ **2026-10-10 收窄**：原口径还有第三处（dsh 插件侧 `dsh/src/el-params.ts` 的
  `mcpPortFile`）—— 那处随**自愈 MCP 桥**一起删了（方案 A：工具面只有一条腿 = 框架的
  聚合端点）⇒ 现在这里反过来钉住"**dsh 侧不许再长出第二条腿**"（反向判断）。
  ⚠ 同时钉住 `ok_codes=()`：**实测** `GET /mcp` 回 406（该端口上无 2xx 路由）⇒ 不许改回默认；
- **C** ⭐ **状态根环境变量已钉死**（§5.6）：`RE0_V2_DATA_ROOT` 必须等于声明的 `data_root`
  （否则换腿后看到另一份 journal，且**无任何报错**）；`EL_NO_AI_SERVER` 必须**存在且为假值**
  （真值会让 MCP 服务端**根本不起**）；`PYTHONIOENCODING` 为 `utf-8`（日志重定向下防 cp936 崩）；
- **D** **CLI 契约已改名**：`app_entry.py` 认 `--mcp`、不再认 `--authority`、无参时指向
  `python -m mecha`（⚠ **不是** `mecha.launch`：那是模块不是包，会静默什么都不做）；
  ⚠ 旧 `launcher.py` **已删**；它在世时 spawn 的也是 `--mcp`。

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
    ('⭐ 声明起的 EL 是 **headless**（argv 里**没有** `--with-gui`）—— 裁决 1 的前提是'
     '「热更新时 GUI 根本不开」，若总是开窗则把当初要避开的代价请回来了',
     _svc is not None and '--with-gui' not in _svc.argv),
    ('data_root 是本仓根（写租约根 = .mecha-v2 的上一层）',
     _svc is not None and Path(_svc.data_root).resolve() == ROOT),
])

# ---------------------------------------------------------------- B. ready 的单一事实
_pf = _svc.ready.port_file if (_svc and _svc.ready) else None
_ok_codes = _svc.ready.ok_codes if (_svc and _svc.ready) else 'N/A'
_v2 = (ROOT / 'src' / 'v2_bind.py').read_text(encoding='utf-8')


def _strip_comments(src: str) -> str:
    """剥掉 TS 注释（`//` 与 `/* */`）。

    ⚠ **必须剥**：下面两条反向判断问的是"**代码**里还有没有桥"，而**注释里解释
    "桥已删/不许回来"是该留的**（否则下一个人查不到为什么不能走老路）。
    本轮**实测踩到**：载体文件的 docstring 里写了 `mountHostPlugin` 三个字 ⇒
    不剥注释就把它自己的判据弄红了（与 EL `dsh/test/re0-panel.test.ts` 同一条教训）。

    ⚠ 不许把 `http://…` 里的 `//` 当注释（`[^:]` 那一位）。
    """
    import re as _re
    src = _re.sub(r'/\*.*?\*/', '', src, flags=_re.S)
    return _re.sub(r'(^|[^:])//.*$', r'\1', src, flags=_re.M)


#: ⭐ **dsh 侧的代码**（剥注释后；用来做反向判断：那条"项目自己接 MCP"的腿**已经删除**，
#: 不许长回来）。⚠ 读**真文件**（R14：gitignore 的产物也要搜；这里读 src 下所有 .ts）。
_dsh_src = _strip_comments('\n'.join(
    p.read_text(encoding='utf-8')
    for p in sorted((ROOT / 'dsh' / 'src').rglob('*.ts'))
))
C.group('B ready：端口文件名是**跨语言单一事实**（软件 ↔ 声明），且 dsh 侧**不再**有第二条腿', [
    ('声明里的 port_file = %r' % _pf, _pf == '.mcp-port'),
    ('Python 侧（src/v2_bind.py）用的是同一个名字',
     _pf is not None and ('"%s"' % _pf) in _v2),
    ('⭐ **反向判断**：dsh 侧不再解析 MCP 端点（`mcpPortFile` / `hubUrl` 一个都不许有）'
     '—— 工具面只有一条腿 = 框架的聚合端点（方案 A，2026-10-10）',
     'mcpPortFile' not in _dsh_src and 'hubUrl' not in _dsh_src),
    ('⭐ **反向判断**：dsh 侧不再 import 自愈桥那套'
     '（`mountHostPlugin` / `@mecha/dsh-panel/host/` 都不许有）',
     'mountHostPlugin' not in _dsh_src and 'dsh-panel/host/' not in _dsh_src),
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
#: ⚠ 旧看门人 `launcher.py` 已删（清理归一 A 类）⇒ 不再读它，改成**反向断言**。
_launcher = ROOT / 'launcher.py'
C.group('D CLI 契约：形态改名成 --mcp，无参指向 python -m mecha', [
    ("app_entry.py 认 '--mcp'", "'--mcp' in _app or '\"--mcp\"' in _app"),
    ("app_entry.py 不再认 '--authority'（契约层清干净）", '--authority' not in _app),
    ('无参时指向 python -m mecha --all，且**不再转发给看门人**（import launcher 已去；'
     '⚠ 注释里那句“不是 mecha.launch”是刻意的提醒，不算违规）',
     'python -m mecha --all' in _app and 'import launcher' not in _app),
    ('⭐ 反向断言：旧看门人 `launcher.py` **已删**（进程编排整块归框架 `python -m mecha`；'
     '本仓不再自己编排、也不再有"两模式交接"）', not _launcher.exists()),
    ("MCP 服务形态的函数是 run_mcp", 'def run_mcp' in _app),
    ('`--with-gui` 仍是**可用能力**（人工/诊断路径没被删；只是声明不带它）',
     '--with-gui' in _app and 'with_gui=' in _app),
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
