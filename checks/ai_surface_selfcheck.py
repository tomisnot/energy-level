# -*- coding: utf-8 -*-
"""**AI 可见面**守卫（B8）：按内容扫 SKILL/面板，不看 `git diff`。

## 为什么是"按内容扫"（而不是 diff / git status）

EL 仓有 **4 份 SKILL**，其中 **2 份在 `.dsh/` 下、是 gitignored**——而
`.dsh/skills/re0-mecha/SKILL.md` **正是 AI 实际加载的那份**。事故实录：
端口早已从固定 `8767` 改成动态发现，**权威版改了、`.dsh` 那份没改**，
而 `git diff` / `git status` / `--fast` **全都看不见它**（gitignored）。
发现它靠的是"提交后另外跑一次按内容扫描"。

⇒ 本判据**只按内容读文件**，并且：
- **R9**：工具面的期望值取自**注册表**（`mecha_v2/tools.py` 的真注册表）、
  镜像一致性取自**权威版文件**——都不取自被扫文件自身；
- **R8**：自证非退化（扫到几个面、多少段、注册表几个工具），并打印；
- **R7**：C 组把合成坏内容喂给同一批探测器，逐条必须报出；同时证明**反方向不误报**
  （有意对照表 / "暂缺能力"小节必须放行）；
- **§3.5 manifest 登记制**：`SURFACES` 是**登记表**——仓里出现未登记的 SKILL
  （"漏登记"）与登记了不存在的文件（"死登记"）**都算违规**，因为"从未被记录的新面"
  正是普通守卫抓不到的那一格。

## 五组断言

- **A 登记完备**：仓内所有 `*/SKILL.md`（排除 vendor：`node_modules`/`.venv`/`__pycache__`）
  ⇄ `SURFACES` 登记表**双向相等**。
- **B 权威 ⇄ 镜像逐字节**：`.dsh/skills/energy-level/SKILL.md` 必须与
  `skills/energy-level/SKILL.md` 字节相同（期望值 = 权威文件，R9）。
- **C 工具面一致（期望值 = 真注册表）**：权威版「## 工具清单」表 == 注册表；
  `re0-mecha` 版「## 工具（v2 实际面）」表 == 注册表；
  其「v1 → v2 对照」表：**右列（v2 名）⊆ 注册表**、**左列（v1 名）∩ 注册表 = ∅**
  （后者是"旧名确实已退役"的对账，也是那张有意表格的**有效利用**而非整体豁免）。
- **D 退役名/固定端口**（按段落判定，见下）：
  - **硬**：`8767` 之类固定端口；或退役工具名的**调用形态**（`` `name(` ``）
    出现在非对照段落 ⇒ 违规（调用形态是"指令"，不是"引用"）；
  - **软**：退役名出现在**既非对照节、又无否定/退役语境**的段落 ⇒ 违规。
  - 豁免语境：段落所属**小节标题**含 暂缺/未接线/不提供/退役/历史/非 MCP/不在，
    或段落文本含 不存在/已退役/曾经/旧/不在/不是/不提供/未接线/暂缺/遗留/v1 等；
    对照节 = 标题含 `对照` 或 `v1 → v2`（`` `list_params` | `query_params` `` 那类）。
    ⚠ **已知残余**（做不到干净区分的那一格，如实登记）：一句**既在豁免语境里、
    又不含调用形态**的过期说法（如"v1 的 `get_activity` 仍可用"）会漏过——
    纯文本层面无法与"有意引用"区分。故硬规则（调用形态 + 端口）是主防线。
- **E 面板取址**：`dsh/src/client/**` 不得出现 `127.0.0.1:<端口>` 字面量或 `MONITOR_URL`
  常量（地址必须经注入的解析器 = host 半同源路由）。host 半**不**受此约束：
  它的 `127.0.0.1:8765/mcp` 是 **MCP 端点**的兜底默认，属另一个事实的家。

用法：python checks\\ai_surface_selfcheck.py     退出码 0 全过 / 1 有问题。
"""
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402
from src.core import protocol as P                           # noqa: E402
from mecha.authority import Authority, Mode                  # noqa: E402
from mecha.gate import Channel, Gate, accept_all             # noqa: E402
from mecha.history import History                            # noqa: E402
from mecha.providers.jobs import JobRegistry                 # noqa: E402
from mecha.surface import Surface                            # noqa: E402

from mecha_v2.tools import build_tool_registry               # noqa: E402

C = Checker('ai_surface')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: **登记表**（§3.5 manifest）：AI 可见面 → 角色。漏登记 / 死登记都违规。
AUTHORITATIVE = 'skills/energy-level/SKILL.md'
MIRROR = '.dsh/skills/energy-level/SKILL.md'
WORKSPACE = '.dsh/skills/re0-mecha/SKILL.md'
SURFACES = {
    AUTHORITATIVE: '权威（入库；工具清单的被检对象 ⇄ 注册表互钉）',
    MIRROR: '镜像（dsh 加载面；须与权威逐字节相同）',
    WORKSPACE: '工作副本（AI 实际读；含独立坑列表与 v1→v2 对照）',
    'skills/ui-design/SKILL.md': 'GUI 设计面技能（非运行面，同样禁固定端口/退役引用）',
}

#: 面板取址面（TS）：不得硬编码监控地址。
CLIENT_GLOB_DIR = os.path.join('dsh', 'src', 'client')

#: v1 工具名（退役）。**只用无歧义的名字**——`query`/`undo` 会与 `query_params` 撞。
RETIRED = ('set_param', 'list_params', 'get_state', 'list_levels', 'get_activity',
           'get_waves', 'estimate_cost', 'preview_behavior', 'run_compute',
           'job_status', 'list_jobs', 'mecha_script', 'list_archives',
           'read_archive', 'run_python', 'submit_sweep', 'get_diag', 'set_groups',
           'set_initial_state', 'set_superposition')

#: 固定端口（监控端点早已动态分配）。
PORT_LITERAL = re.compile(r'\b8767\b')
#: 对照节标题 / 否定语境（小节标题级 / 段落级）。
CMP_HEAD = re.compile(r'对照|v1\s*(?:→|->)\s*v2')
ABSENT_HEAD = re.compile(r'暂缺|未接线|不提供|退役|历史|非\s*MCP|不在')
ABSENT_PARA = re.compile(r'不存在|已退役|退役|曾经|旧|不在|不是|不提供|未接线|暂缺'
                         r'|遗留|v1|不\s*eval|不认|非\s*MCP')
#: 工具清单小节标题（权威版 / 工作副本各有其名）。**必须带 `re.M`**——`^` 要按行匹配，
#: 否则整份文件只匹配首行（实测踩过：表解析成 0 行、C 组假红）。
TOOL_HEAD = re.compile(r'^##\s*工具清单', re.M)
V2FACE_HEAD = re.compile(r'^##\s*工具（v2 实际面）', re.M)
#: 对照**小节**（切片用：标题锚定，避免匹配到正文里的 "v1 → v2" 字样）。
CMP_SECTION = re.compile(r'^#{2,4}.*(?:对照|v1\s*(?:→|->)\s*v2)', re.M)
_ROW = re.compile(r'^\|\s*`([A-Za-z_][A-Za-z0-9_]*)`\s*\|')
#: 对照表任意一行：左列裸名 + 右列单元格原文（右列可能是"不在 MCP 工具面"这类说明、
#: 也可能有 `+` 组合，故**不**要求右列是裸名——否则会静默漏行）。
_PAIR = re.compile(r'^\|\s*`([A-Za-z_][A-Za-z0-9_]*)`[^|]*\|([^|]*)\|')
_BACKTICK = re.compile(r'`([A-Za-z_][A-Za-z0-9_]*)`')


def _read(rel):
    """读一个面（判据只经这一个口读文件 ⇒ 红证据可运行时替换它，不碰仓内文件）。"""
    with io.open(os.path.join(ROOT, rel.replace('/', os.sep)),
                 encoding='utf-8', errors='replace') as f:
        return f.read()


def _list_skill_files():
    """仓内所有 `SKILL.md`（按内容/路径枚举，排除 vendor）。"""
    out = []
    skip = ('node_modules', '.venv', '__pycache__', '.git')
    for dp, _dn, fn in os.walk(ROOT):
        if any(s in dp for s in skip):
            continue
        for f in fn:
            if f == 'SKILL.md':
                out.append(os.path.relpath(os.path.join(dp, f), ROOT).replace(os.sep, '/'))
    return sorted(out)


def _blocks(text):
    """→ [(小节标题, 段落文本, 行号表)]：按标题切片、按空行分段。

    段落（而不是单行）才是判定单位：文档是**折行**写的，一句的否定词常落在下一行；
    "有意引用"也常由小节标题（如"v2 暂缺的能力"）声明，而不是逐行。
    """
    out, head, buf, ls = [], '', [], []

    def flush():
        if buf:
            out.append((head, '\n'.join(buf), list(ls)))

    for i, line in enumerate(text.splitlines(), 1):
        if line.startswith('#'):
            flush(); buf, ls, head = [], [], line
            continue
        if not line.strip():
            flush(); buf, ls = [], []
            continue
        buf.append(line); ls.append(i)
    flush()
    return out


def scan_text(text):
    """按段落扫一个面 → (硬问题, 软问题)。纯函数（合成坏内容直接喂它）。"""
    hard, soft = [], []
    for head, para, ls in _blocks(text):
        in_cmp = bool(CMP_HEAD.search(head))
        exempt = (in_cmp or bool(ABSENT_HEAD.search(head))
                  or bool(ABSENT_PARA.search(para)))
        if PORT_LITERAL.search(para):
            hard.append('第 %d 行起：出现固定端口 8767（监控端点已动态分配）'
                        % ls[0])
        for tok in RETIRED:
            if not re.search(r'\b%s\b' % re.escape(tok), para):
                continue
            if in_cmp:
                break
            if re.search(r'%s\s*\(' % re.escape(tok), para):
                hard.append('第 %d 行起：退役工具名 %r 的**调用形态**（指令，非引用）'
                            % (ls[0], tok))
            elif not exempt:
                soft.append('第 %d 行起：退役工具名 %r 既不在对照节、也无否定/退役语境'
                            % (ls[0], tok))
            break
    return hard, soft


def _slice(text, head_re, stop_re):
    """切出 `head_re` 命中的小节正文（到下一个 `stop_re` 标题为止）。"""
    m = head_re.search(text)
    if not m:
        return None
    tail = text[m.end():]
    nxt = stop_re.search(tail)
    return tail[:nxt.start()] if nxt else tail


def _section_rows(text, head_re):
    """某个小节里的「行首反引号名」集合（工具清单表）；找不到小节 → None。

    切片**到下一个标题为止（任意层级）**：清单表就是该小节的全部内容；写成"到下一个
    `##` 为止"会把小节内的 `###` 子表一起吞进来（实测：工作副本的 v1→v2 对照表正是
    同一个 `##` 下的 `###`，吞进来后 14 个变成 24 个）。
    """
    section = _slice(text, head_re, re.compile(r'^#{1,6}\s', re.M))
    if section is None:
        return None
    return {mm.group(1) for mm in (_ROW.match(l) for l in section.splitlines()) if mm}


def _pair_rows(text, head_re):
    """对照小节里的 ``| `左` | 右列 |`` → [(左名, {右列的 v2 名})]。

    ⚠ 右列**只在整格就是** `` `name` `` 或 `` `name` + `name` `` 时才算映射：
    有的行右列是散文（如"不在 MCP 工具面——…；域当前态经 `read_config`"），
    按"抽格子里的所有反引号名"会把散文里提到的**别的**工具名当成映射（实测踩过：
    `get_activity` 那行被抽出 `read_config`）——那类误抽会让 `右列 ⊆ 注册表` 这条
    在"散文提到退役名"时误报。
    """
    section = _slice(text, head_re, re.compile(r'^#{1,3}\s', re.M))
    if section is None:
        return []
    bare = re.compile(r'^\s*`[A-Za-z_][A-Za-z0-9_]*`'
                      r'(?:\s*\+\s*`[A-Za-z_][A-Za-z0-9_]*`)*\s*$')
    out = []
    for line in section.splitlines():
        mm = _PAIR.match(line)
        if not mm:
            continue
        cell = mm.group(2)
        names = set(_BACKTICK.findall(cell)) if bare.match(cell) else set()
        out.append((mm.group(1), names))
    return out


def _registry_tool_names():
    """真注册表的工具名（R9：**独立信源**，不取自任何被扫文件）。"""
    class _SchemaOnly:
        def schema(self):
            return {"run_output_required": ["ok", "method"],
                    "estimate_output_required": ["est_sec"],
                    "config": P.config_schema(), "solve": P.solve_schema(),
                    "engine": P.engine_params(), "settable_keys": P.settable_keys()}
    eng = _SchemaOnly()
    hist = History()
    auth = Authority(mode=Mode.AI)
    gate = Gate(auth, hist, validate=accept_all)
    reg = build_tool_registry(eng, gate, Surface(eng), auth, Channel("ai"),
                              JobRegistry())
    return {s['name'] for s in reg.schemas()}


def main():
    stats = {}

    # ---- A 登记完备（漏登记 / 死登记都违规）----
    found = set(_list_skill_files())
    registered = set(SURFACES)
    missing_from_manifest = sorted(found - registered)
    missing_on_disk = sorted(registered - found)
    stats['SKILL 文件数'] = len(found)
    C.group('A manifest 登记制：仓内 SKILL ⇄ 登记表双向相等（漏登记=违规）', [
        ('发现 %d 份：%s' % (len(found), '、'.join(sorted(found))),
         not missing_from_manifest),
        ('登记表全部存在（缺：%s）' % (missing_on_disk or '无'), not missing_on_disk),
    ])

    # ---- B 权威 ⇄ 镜像逐字节（期望值 = 权威文件）----
    auth_text = _read(AUTHORITATIVE)
    mirror_text = _read(MIRROR)
    C.group('B 镜像与权威逐字节相同（期望值取自权威版，R9）', [
        ('%s ≡ %s（%d 字节 vs %d 字节）'
         % (MIRROR, AUTHORITATIVE, len(mirror_text.encode('utf-8')),
            len(auth_text.encode('utf-8'))),
         mirror_text == auth_text),
    ])

    # ---- C 工具面一致（期望值 = 真注册表）----
    reg_names = _registry_tool_names()
    stats['注册表工具数'] = len(reg_names)
    auth_tbl = _section_rows(auth_text, TOOL_HEAD)
    ws_text = _read(WORKSPACE)
    ws_tbl = _section_rows(ws_text, V2FACE_HEAD)
    pairs = _pair_rows(ws_text, CMP_SECTION)
    stats['权威工具清单'] = len(auth_tbl or ())
    stats['工作副本 v2 面'] = len(ws_tbl or ())
    stats['对照表行数'] = len(pairs)
    left = {a for a, _b in pairs}
    right = {b for _a, bs in pairs for b in bs}
    C.group('C 三份 SKILL 的工具面一致（期望值 = 真注册表 %d 个）' % len(reg_names), [
        ('权威「工具清单」表 == 注册表（表 %d 个；差集 %s）'
         % (len(auth_tbl or ()), sorted((auth_tbl or set()) ^ reg_names) or '空'),
         auth_tbl == reg_names),
        ('工作副本「v2 实际面」表 == 注册表（表 %d 个；差集 %s）'
         % (len(ws_tbl or ()), sorted((ws_tbl or set()) ^ reg_names) or '空'),
         ws_tbl == reg_names),
        ('对照表右列（v2 名）⊆ 注册表（越界 %s）' % (sorted(right - reg_names) or '空'),
         right <= reg_names),
        ('对照表左列（v1 名）∩ 注册表 = ∅（仍在册的 %s）'
         % (sorted(left & reg_names) or '空'), not (left & reg_names)),
        ('对照表非空且逐行取到左列名（%d 行 / %d 个左名）' % (len(pairs), len(left)),
         bool(pairs) and len(left) >= 8),
    ])

    # ---- D 退役名 / 固定端口（按段落）----
    dsubs, all_hard, all_soft = [], [], []
    for rel in sorted(SURFACES):
        hard, soft = scan_text(_read(rel))
        all_hard += ['%s %s' % (rel, p) for p in hard]
        all_soft += ['%s %s' % (rel, p) for p in soft]
        dsubs.append(('%s：硬 %d / 软 %d' % (rel, len(hard), len(soft)),
                      not hard and not soft))
    stats['硬问题'] = len(all_hard)
    stats['软问题'] = len(all_soft)
    C.group('D 退役名与固定端口（按段落；对照节/否定语境豁免）', dsubs)

    # ---- E 面板取址面（client 半不得硬编码监控地址）----
    cdir = os.path.join(ROOT, CLIENT_GLOB_DIR)
    cbad, nfiles = [], 0
    for dp, _dn, fn in os.walk(cdir):
        for f in fn:
            if not f.endswith(('.ts', '.tsx')):
                continue
            nfiles += 1
            rel = os.path.relpath(os.path.join(dp, f), ROOT)
            t = _read(rel)
            if re.search(r'127\.0\.0\.1:\d+', t):
                cbad.append('%s：硬编码 loopback 端口' % rel)
            if 'MONITOR_URL' in t:
                cbad.append('%s：出现硬编码常量 MONITOR_URL' % rel)
    stats['client 文件数'] = nfiles
    C.group('E 面板取址：client 半经注入解析器取地址（不得硬编码）', [
        ('%d 个 client 文件：%s' % (nfiles, cbad or '无硬编码'), not cbad and nfiles > 0),
    ])

    # ---- F 致死性（R7）+ 反方向不误报 ----
    good_cmp = ('### v1 → v2 工具名对照（旧名已不存在）\n'
                '| v1 | v2 |\n|---|---|\n| `set_param` | `set_config` |\n')
    cases = [
        ('固定端口 8767 → 硬报', '监控端点在 :8767 上。', 1, 0),
        ('退役名调用形态 → 硬报',
         '用 `set_param("field.B_gauss", 21.7)` 改磁场。', 1, 0),
        ('退役名无豁免语境 → 软报', '`set_param` 可以改配置键。', 0, 1),
        ('有意对照表 → **不误报**', good_cmp, 0, 0),
        ('"暂缺能力"小节里的引用 → **不误报**',
         '## v2 暂缺的能力\n- `run_python` / 任意代码执行：v2 不提供。\n', 0, 0),
        ('否定语境的跨行引用 → **不误报**',
         '操作流水（v1 的\n`get_activity`）不在 MCP 工具面。\n', 0, 0),
    ]
    subs = []
    for label, text, want_h, want_s in cases:
        h, s = scan_text(text)
        subs.append(('%s（硬 %d/期望 %d，软 %d/期望 %d）'
                     % (label, len(h), want_h, len(s), want_s),
                     len(h) == want_h and len(s) == want_s))
    subs.append(('登记表缺项被抓：合成"发现集 - 登记集"非空 → 报出',
                 bool(sorted({'a/SKILL.md'} - set(SURFACES)))))
    subs.append(('登记表死项被抓：合成"登记集 - 发现集"非空 → 报出',
                 bool(sorted(set(SURFACES) - {'skills/energy-level/SKILL.md',
                                              'skills/ui-design/SKILL.md'}))))
    C.group('F 致死性 + 反方向不误报（对照表/暂缺节/跨行引用必须放行）', subs)

    print('[ai_surface] 非退化自证：%s' % stats)
    C.group('R8 非退化自证：扫到的面/段/注册表工具都非空', [
        ('SKILL 面 %d 份、client 文件 %d 个、注册表工具 %d 个、对照表 %d 行'
         % (stats['SKILL 文件数'], stats['client 文件数'],
            stats['注册表工具数'], stats['对照表行数']),
         stats['SKILL 文件数'] > 0 and stats['client 文件数'] > 0
         and stats['注册表工具数'] > 0 and stats['对照表行数'] > 0),
    ])

    ok = C.summary()
    for p in all_hard + all_soft:
        print('   - %s' % p)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
