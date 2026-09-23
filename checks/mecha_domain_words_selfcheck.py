# -*- coding: utf-8 -*-
"""领域词表守卫（P1-9h 收口）——「库里出现具体领域值就是未凝练」的**致死判据**。

## 这条判据守什么

库（`mecha-sdk`）的提炼边界纪律（`docs/审计-mecha-sdk领域凝练.md` §0/§3）：

> 库里出现**具体值**（键名/动词名/物理量/项目名/领域目录名）就是**未凝练**——
> 应改成「库声明接口、项目侧提供值」。在库源码上 grep 一张**领域词表**，
> 真代码命中必须为 **0**（注释与 docstring 除外，`examples/` 不扫）。

2026-09-22 主代理用 AST 剥掉注释与 docstring 复核：直接 grep ~60 处命中，
剥干净后真代码只剩 3 个**族**（`save_state` / `list_levels` / `runspec`）。
本守卫把那次手工复核固化成判据——否则同样的残留下次还能静默活下去。

## 判据结构（A 事实 / B 前提 / C 探测器载重 / D 白名单不静默）

  A. **事实**：扫库真代码，非白名单命中必须 0；白名单放行了哪些**逐条列出**
     （不藏）；每个白名单条目必须**仍命中**（0 命中 ⇒「该清理白名单了」——腐烂侦测）。
  B. **前提**：能定位库（editable 安装的 `mecha` + 平级 `cockpit` 两个包都扫）；
     扫描面与词表写死在本文件里（改词表 = 改判据，必须显式）。
  C. **探测器载重**（致死性 + 不许误报，同一探测器喂合成源）：
     · 合成源塞 `'rabi_mhz'`（字符串常量）与 `two_body = True`（标识符）⇒ **必须红**；
     · 干净的通用源（`log_level`/`autonomy_level`/`top_level`/`'level'` 告警级/
       `state`/`mode` 等框架词）⇒ **必须 0 命中**（证明不是永远返空的哑守卫）；
     · 领域词**只**出现在注释/docstring 里的合成源 ⇒ 0 命中（证明剥干净是真的）。
  D. **白名单不静默**：每条必须带**理由**（跨仓在用/库自己的产物名…），
     空理由 = 判据不过（防止"为了绿悄悄放行"）。

## 边界（诚实声明，避免守卫假装万能）

- 只看**词表面**：它抓"领域词汇漏进库"，抓不了"库替领域解释结构"（那需要
  P0-5/P1-9c 的键契约对拍，如 `ops` 读 preview 的领域键——由各自机制守）。
- **词表是子串匹配、大小写敏感**：候选词来自审计 §3（可增删，但必须写死在本
  文件并随判据一起评审）。**故意不收**裸 `level`/`state`/`pair`/`mode`——它们
  命中 `log_level`/`autonomy_level`/`top_level`/告警 `'level'`/`Ops.state`/
  运行 `mode` 等**框架词汇**，收进来只会误报、把真信号淹掉。领域形状的是
  `list_levels` 这个名字本身（它假定"可寻址对象=能级"）——单独收。
- **不扫 `examples/`**（审计 §3 明指的例外；它们本来也不在 `mecha/`+`cockpit/`
  两个包内，天然排除）。

纯 AST + 轻量 import，不跑引擎、不写任何文件（秒级）。
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = Checker('mecha_domain_words')

# ---------------------------------------------------------------- 定位库
# 库已外移（editable 安装）；本守卫扫的是**库源码**（mecha/ + cockpit/ 两个包），
# 不是本仓。找不到 ⇒ 明确 FAIL/SKIP 并说明（不静默通过——那正是 D4 假绿）。
try:
    import mecha as _mecha_pkg
    _LIB = os.path.dirname(os.path.dirname(os.path.abspath(_mecha_pkg.__file__)))
except Exception as e:                                   # noqa: BLE001
    C.skip('领域词表守卫', 'mecha-sdk 不在（%s: %s）' % (type(e).__name__, e))
    sys.exit(0 if C.summary() else 1)

_PKGS = [os.path.join(_LIB, 'mecha'), os.path.join(_LIB, 'cockpit')]

# ---------------------------------------------------------------- 词表（写死）
# 来源：docs/审计-mecha-sdk领域凝练.md §3 的候选表 + P1-9h 工单；大小写敏感子串。
DOMAIN_WORDS = (
    'wave',            # 波形族（wave_list/waves_*/wavelength…）
    'rabi',            # 拉比频率（rabi_mhz/rabi_of…）
    'detun',           # 失谐（detuning_*）
    'gauss',           # 磁场单位方言（B_gauss）
    'Re0',             # 项目名（大小写敏感：这是项目身份，不是通用词）
    'save_state',      # EL 的存档目录名（P1-9f 的靶子）
    'runspec',         # EL 内部模块名（库产物格式名族的来源）
    'psi_',            # 初态键族（psi_single/psi_pair…）
    'two_body',        # 两体方言
    'n_waves',         # 波形条数旁注键
    'overlap',         # 波形重叠告警
    'atom',            # 原子族（n_atoms/atom_list…）
    'rydberg',         # 里德堡
    'arrhenius',       # 动力学方言
    'boltzmann',       # 统计物理方言
    'drive',           # 驱动族（set_drive/driven/driver…）
    'energy',          # 能量方言
    'list_levels',     # "可寻址对象=能级"的旧名入口（领域形状的方法名）
)

# ---------------------------------------------------------------- 白名单（带理由）
# 放行 ≠ 干净：每条**必须**带理由，且带腐烂侦测（A 组：0 命中 ⇒ 提示该清理）。
# 判据自身不允许"为了绿把词从表里删掉"——要收窄只能改这张表并随判据评审。
WHITELIST = {
    'list_levels': (
        '跨仓在用的旧名入口：库（server/ops/protocol/codeact）与主仓 '
        '（adapter、Hub 门面 GUI 取态、SKILL/README 守卫）都钉着它。'
        'MCP 工具 list_levels ≡ query("list_levels")（P1-6 已给等价入口）。'
        '彻底移除需排期统一删：移除清单见 mecha-sdk 提交说明与 '
        'docs/设计-静态能力AI入口.md §10。'
        '**不删白名单、不静默放行**：它存在一天，这张表就记一天。'),
    'runspec': (
        '库**自己的产物格式名族**（runspec.pkl / runspec.json / '
        'runspec_format / runspec_json_note，D8 可复现产物的文件名与 '
        'manifest 键）+ codeact.NOT_EXPOSED 里的框架级裸入口名。'
        '判据：这些是库产物契约（任何领域都有"可复现产物"），不是某个领域的'
        '业务词；但名字源自 EL 内部模块名，故登记在案。'
        '中性化方向：文件名改 run.pkl / run.json（改时同步删本白名单条目'),
}


def _py_files(dirs):
    """两个包的**全部** .py（os.walk：子包也覆盖，别只列顶层）。"""
    out = []
    for d in dirs:
        for dp, _dn, fn in os.walk(d):
            out.extend(os.path.join(dp, f) for f in fn if f.endswith('.py'))
    return sorted(out)


# ★ 非 .py 的**文本资产**也要扫（D-1 的教训）：`cockpit/static/` 是出厂 UI，
#   里面有标题/文案/前端键名——旧版只扫 `.py` ⇒ `Re0` 明明在词表里却**扫不到**，
#   于是 `index.html` 的 `<title>Re0 机甲驾驶舱…</title>` 一路漏到审查者手里。
#   同类问题换个扩展名就会再来一次，故扫描面必须按**资产类型**收，而不是按扩展名。
_TEXT_ASSET_EXT = ('.html', '.js', '.css', '.json', '.md', '.txt', '.svg', '.yml', '.yaml')


def _text_assets(dirs):
    """两个包里的**非 .py 文本资产**（出厂 UI/静态资源/随包文档）。"""
    out = []
    for d in dirs:
        for dp, _dn, fns in os.walk(d):
            for fn in fns:
                if fn.endswith('.py'):
                    continue
                if fn.lower().endswith(_TEXT_ASSET_EXT):
                    out.append(os.path.join(dp, fn))
    return sorted(out)


def scan_text(src, path='<synthetic>'):
    """**非 .py 文本** → 命中列表（逐行子串匹配；无 AST，故**注释也算**——
    对出厂 UI 而言注释与文案同等重要，都会被新项目照抄）。"""
    hits = []
    for i, line in enumerate(src.split('\n'), 1):
        for w in DOMAIN_WORDS:
            if w in line:
                hits.append((path, i, w, line.strip()[:60], 'text'))
    return hits


def _strip_docstrings(tree):
    """剥掉模块/类/函数的 docstring（注释本就不在 AST 里，天然排除）。"""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _iter_code_tokens(tree):
    """→ [(lineno, token, kind)]：真代码里的**命名**与**字符串常量**。

    两者都算：`DEFAULT_DIR = 'save_state'`（值泄漏）与
    `def list_levels(...)`（命名泄漏）是同一族未凝练。
    """
    for nd in ast.walk(tree):
        names = []
        if isinstance(nd, ast.Name):
            names.append(nd.id)
        elif isinstance(nd, ast.Attribute):
            names.append(nd.attr)
        elif isinstance(nd, ast.arg):
            names.append(nd.arg)
        elif isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(nd.name)
        elif isinstance(nd, ast.keyword):
            names.append(nd.arg or '')
        for nm in names:
            yield (nd.lineno, nm, 'name')
        if isinstance(nd, ast.Constant) and isinstance(nd.value, str):
            yield (nd.lineno, nd.value, 'str')


def scan_source(src, path='<synthetic>'):
    """一段源码 → [(path, lineno, word, token, kind)]（词表命中，未过白名单）。"""
    tree = _strip_docstrings(ast.parse(src))
    hits = []
    for ln, tok, kind in _iter_code_tokens(tree):
        for w in DOMAIN_WORDS:
            if w in tok:
                hits.append((path, ln, w, tok[:60], kind))
    return hits


# ---------------------------------------------------------------- A. 扫真库
_files = _py_files(_PKGS)
_raw_hits = []                    # 全量命中（过白名单前）
for _p in _files:
    with open(_p, encoding='utf-8') as f:
        try:
            _raw_hits.extend(scan_source(f.read(), _p.replace(_LIB + os.sep, '')))
        except SyntaxError as e:                        # noqa: BLE001
            _raw_hits.append(('<parse-error>', e.lineno or 0, e.msg, '', 'err'))

_residual = [h for h in _raw_hits if h[2] not in WHITELIST]
_whitelisted = [h for h in _raw_hits if h[2] in WHITELIST]
_wl_counts = {w: sum(1 for h in _raw_hits if h[2] == w) for w in WHITELIST}

# ---------------------------------------------------------------- A2. 扫非 .py 文本资产（D-1）
_assets = _text_assets(_PKGS)
_asset_hits = []
for _p in _assets:
    with open(_p, encoding='utf-8', errors='replace') as f:
        _asset_hits.extend(scan_text(f.read(), _p.replace(_LIB + os.sep, '')))
_asset_residual = [h for h in _asset_hits if h[2] not in WHITELIST]


def _fmt(hits, limit=8):
    out = ['%s:%d[%s] %s→%r' % (os.path.basename(h[0]), h[1], h[4], h[2], h[3])
           for h in hits[:limit]]
    if len(hits) > limit:
        out.append('…共 %d 处' % len(hits))
    return '; '.join(out)


print('库：%s（扫 mecha/ + cockpit/ 共 %d 个 .py + %d 个非 .py 文本资产；'
      'examples/ 不在两包内故天然排除）' % (_LIB, len(_files), len(_assets)))
print('词表 %d 个：%s' % (len(DOMAIN_WORDS), ' '.join(DOMAIN_WORDS)))
if _whitelisted:
    print('白名单放行 %d 处：%s' % (len(_whitelisted), _fmt(_whitelisted)))
    for _w, _why in WHITELIST.items():
        print('  · [%s ×%d] 理由：%s' % (_w, _wl_counts.get(_w, 0), _why))

C.group('A 库真代码领域词命中（剥注释/docstring 后；白名单放行 %d 处）'
        % len(_whitelisted), [
    ('**非白名单命中 0 处**（命中：%s）' % (_fmt(_residual) or '无'), not _residual),
] + [('**腐烂侦测**：白名单条目 %r 仍命中 %d 处（0 ⇒ 该清理白名单了——'
      '说明它已修好/已移除，放行理由失效）' % (w, _wl_counts.get(w, 0)),
      _wl_counts.get(w, 0) > 0) for w in WHITELIST])

# ★ A2：**非 .py 文本资产**（D-1 的教训）。出厂 UI 也是"会被照抄的形状"，
#   旧版只扫 .py ⇒ `index.html` 的 `<title>Re0 机甲驾驶舱…</title>` 一路漏掉。
C.group('A2 **非 .py 文本资产**领域词命中（%d 个文件：static 前端/随包文档）'
        % len(_assets), [
    ('扫描面含非 .py 资产（%d 个）' % len(_assets), len(_assets) >= 1),
    ('**非白名单命中 0 处**（命中：%s）' % (_fmt(_asset_residual) or '无'),
     not _asset_residual),
    ('**致死性**：合成一段含 `Re0` 的 HTML ⇒ 必被 `scan_text` 检出（非哑守卫）',
     bool(scan_text('<title>Re0 机甲驾驶舱</title>', 'synthetic.html'))),
    ('**不许误报**：中性 HTML ⇒ 0 命中',
     not scan_text('<title>mecha 驾驶舱</title>', 'synthetic.html')),
])

C.group('B 前提：定位到库 + 扫描面正确（mecha/ + cockpit/；docstring 已剥）', [
    ('库目录（%s）含 mecha/ 与 cockpit/ 两个包'
     % os.path.basename(_LIB),
     os.path.isdir(os.path.join(_LIB, 'mecha'))
     and os.path.isdir(os.path.join(_LIB, 'cockpit'))),
    ('扫到 %d 个 .py（≥20 才算真扫了库，防"目录没了当空过"）' % len(_files),
     len(_files) >= 20),
    ('cockpit/ 的 3 个文件在扫描面内（黑匣子也是库的一部分）',
     any(f.endswith(os.path.join('cockpit', 'backend.py')) for f in _files)
     and any(f.endswith(os.path.join('cockpit', 'server.py')) for f in _files)
     and any(f.endswith(os.path.join('cockpit', '__init__.py')) for f in _files)),
])

# ---------------------------------------------------------------- C. 致死性 + 不许误报
# 合成源与 mecha_purity 的 D 组同思路：同一探测器必须分得清好坏。
_rogue_str = "# a comment mentions rabi_mhz\n" \
             "CONST = 'rabi_mhz'\n" \
             "two_body = True\n" \
             "def helper(save_state):\n    return save_state\n"
_rogue_doc = '"""docstring 里写 rabi_mhz 与 save_state 都不算（剥掉）。"""\n' \
             "def f(x):\n    'rabi_mhz in a non-first string stays'\n" \
             "    return x\n"
_clean = ("from . import runtime\n"
          "import os, json\n"
          "class Gate:\n"
          "    log_level = 'INFO'\n"
          "    def autonomy_level(self, op):\n        return 1\n"
          "    def state(self, top_level=True):\n"
          "        return {'level': 'warn', 'mode': 'ai', 'states': []}\n"
          "def set_param(key, value, reason):\n    return runtime.engine()\n")
_lethal_hits = scan_source(_rogue_str)
_lethal_words = {h[2] for h in _lethal_hits}
_doc_hits = scan_source(_rogue_doc)
_clean_hits = scan_source(_clean)
C.group('C 致死性 + 不许误报（同一探测器喂合成源；非哑守卫）', [
    ("**致死性**：塞一行 'rabi_mhz'（字符串常量）必须变红",
     'rabi' in _lethal_words),
    ('**致死性**：标识符 `two_body = True` 必须变红',
     'two_body' in _lethal_words),
    ('**致死性**：形参名 `save_state` 必须变红（命名泄漏同族）',
     'save_state' in _lethal_words),
    ('docstring 里的领域词被剥掉 ⇒ 0 命中（"剥干净"是真的）',
     not _doc_hits),
    ('**不许误报**：干净通用源（log_level/autonomy_level/top_level/告警 level/'
     'state/mode…）0 命中', not _clean_hits),
])

# ---------------------------------------------------------------- D. 白名单不静默
C.group('D 白名单每条带理由（防止"为了绿悄悄放行"）', [
    ('白名单 %d 条，理由均非空' % len(WHITELIST),
     len(WHITELIST) >= 1 and all(isinstance(r, str) and len(r) >= 10
                                 for r in WHITELIST.values())),
    ('list_levels 条目写明等价入口与移除流程', '移除' in WHITELIST['list_levels']),
    ('runspec 条目写明"库自己的产物名"判据与中性化方向',
     '产物' in WHITELIST['runspec']),
])

_ok = C.summary()
if __name__ == '__main__':
    sys.exit(0 if _ok else 1)
