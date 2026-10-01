# -*- coding: utf-8 -*-
"""dsh 面板**共享资产**守卫（EL 侧）：常量指纹 + 共享单测真跑。

## 它守什么

`dsh/src/dsh-panel/` 是 mecha 参考实现 `mecha/dsh-panel/` 的**逐字副本**——按该目录
`README.md` 的复制约定：**只有参数块 `panel-config.ts` 允许项目化**（EL 的路由 / 端口
文件名 / 标题），**其余文件必须与参考实现逐字一致**。出处（含本仓抄了哪两批件）见下面的
`REFERENCE_REVISION`——**别在这段里另抄一个提交哈希**（D1：事实一个家）。

2026-10-01 起本目录含**两类**标准件（资产 README 的分类）：

* **默认件**（面板 + 地址路由）：`routes.ts` / `panel-view.ts` / `panel-data.ts` /
  `MonitorTabBody.tsx` / `monitor-client.ts` / `monitor-url.ts` + 各自的 `.test.ts`；
* **opt-in 件**（自愈 MCP 桥 + 工具注册）：`mcp-bridge.ts` / `mcp-session-http.ts` /
  `config.ts` / `register-tools.ts` / `mcp-sdk-shims.d.ts` + `mcp-bridge.test.ts`
  ——EL **要用**（dsh 里的 AI 原生调自家工具），故照抄；**它们的项目值经参数传进去**，
  见 `dsh/src/el-params.ts` 与 `dsh/src/host/el-hub.ts`（这两个文件**不是**资产件，不进本表）。

本判据把**行为载体**（代码 + 单测）的**行尾归一 sha256** 钉成常量：

* **优先「钉常量指纹」，运行时零跨仓读取**（资产 README 的裁决）：上游任何在飞改动都
  不会红本判据；上游真发新版时**整份重新复制 + 同步更新常量**，而不是就地打补丁。
  `REFERENCE_REVISION` 只作**出处说明**。
* ⚠ 比的是**归一化行尾后的正文**：`core.autocrlf` 会让"同一内容"字节不同（实测过）
  ⇒ `\\r\\n` 归一成 `\\n` 再取指纹；裸字节比对会被行尾配置打败。
* ⚠ **参数块与 README.md 不进表**：前者必然不同（它就是参数块），后者是 prose、
  就地批注不改行为（钉它只制造噪声）。
* ⚠ 本判据**按内容读文件、不经 git**：本工程两次教训——gitignore 的文件逃出
  `git diff` 验收；目录 junction 让"两条路径"其实是同一个文件。

## 三组断言

- **A 指纹**：逐行为载体存在 + 归一后 sha256 == 常量（不符即"漂移"）。
- **B 致死性（R7）**：把合成内容喂给**同一个**探测函数——改一个字符 / 少一个文件 /
  干净内容——前两者必须报、后者必须零问题（否则它是哑守卫）。
- **C 共享单测真跑**：`node --test` 跑表里的共享单测，解析 **tests/pass/fail/skipped**
  四个计数，要求 **`pass > 0` 且 `skipped == 0`**（R16 实例：本工程有一次判据**漏了
  `skipped`**，把"全 skip"判成 PASS ⇒ "全被跳过"是红，不是绿）。

## 非退化自证（R8）

打印并断言：钉了几个载体、哈希了多少字节、共享单测跑出几个用例。否则"表是空的"
也会"看起来通过"。

用法：python checks\\dsh_panel_asset_selfcheck.py     退出码 0 全过 / 1 有问题。
"""
import hashlib
import io
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402

C = Checker('dsh_panel_asset')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DIR = os.path.join('dsh', 'src', 'dsh-panel')

#: **出处说明**（只作说明，不参与比对——比对靠下面那组常量）。
REFERENCE_REVISION = ('mecha/dsh-panel @ c0cd1d3（2026-10-01：默认 10 件仍与 7490db2 逐字相同，'
                      '本批新增 **opt-in 件** mcp-bridge / mcp-session-http / config / '
                      'register-tools / mcp-sdk-shims + mcp-bridge.test，README 同步；'
                      '更早 7490db2、58e2f8a、f516a99、c44b01f）')

#: **行为载体**（代码 + 单测）→ 行尾归一 sha256。改这里 = "上游发新版后整份重新复制"。
#: ⚠ `panel-config.ts` 不进表（它是参数块，必然不同）；`README.md` 不进表（prose）。
#: ✅ `panel-data.ts` / `panel-data.test.ts` 已随上游 `5ee78e4` 重抄（上游修了两处**只有消费者
#: 才看得见**的缺陷：① `Ev`/`ConfigWire` 漏 `import type`，在严格 tsconfig 下 TS2304；
#: ② `tabLabel` 用例把共享默认写死、却读项目参数块 ⇒ 照文档填 `TABS` 的项目必红）。
#: ⚠ 教训：**资产自测只剥类型、不做类型检查**，所以那类缺陷**只有消费者的 `npm run typecheck` 能抓**；
#: 上游现已把类型检查纳入它自己的门禁。
CARRIERS = {
    # 浏览器安全的共享常量（零 import）——client 半的 import 闭包靠它避开 node:*
    'routes.ts':
        'e2b5bc35d84d368548146a4c7d6c4a2ef733b3191480f6a9d0131045c1295963',
    # node-only（host 半）
    'monitor-url.ts':
        '9808f71a7179857f56baa25652cfc77e2d109f3d92d643aaf2c3ebc42f95c2a3',
    # client 半：取数骨架 + 数据层 + 纯字符串视图 + 薄壳（唯一需要 react 的地方）
    'monitor-client.ts':
        '200a7764f2d0139c17dd91294d02dd165ad0f51ad782ad39b3bb67708153e93c',
    'panel-data.ts':
        'cc32f9c7692fdbf2f0495d23f2f4bab2ac3e27620795e92f734132c96a0153bc',
    'panel-view.ts':
        '560103c18231df5bad3d18a77f16f0ed6cceab513928ef765dbdb2afec9a1aca',
    'MonitorTabBody.tsx':
        '8ed1233f18e32a9c4b6af37ebd69937fe63f5522ce2a410bb0dab8ec9339d3d6',
    # opt-in 件（2026-10-01 批次）：自愈 MCP 桥 + 端点解析 + 工具注册 + SDK 类型垫片。
    # ⚠ 它们的**项目值经参数传进去**（`dsh/src/el-params.ts` / `host/el-hub.ts`），
    #   所以这些文件本身逐字一致 = 合同；参数填错会在 EL 自己的验收测试里红。
    'mcp-bridge.ts':
        '625fdd325a81635fc7fac1e55c7358c939f40fc9220a18300b8ed5b99b38ace8',
    'mcp-session-http.ts':
        '423dca6f422491690e91c03b1576296b578162aa5be72684419bc510445fa194',
    'config.ts':
        '537668e2c0b4af8bc0b624ec42cd9e5c0f126205ab676e7700bc592e137c4207',
    'register-tools.ts':
        '4eadbf2f495f2f16e09a197db920b1662cb17ddbf051f563509fbe08e0174bf0',
    'mcp-sdk-shims.d.ts':
        'd2fc24f0d7131fa2c2e700e9d669c51578f0698023c7919b3b3d36f027aa331e',
    # 单测（随实现一起复制；由 C 组真跑）
    'monitor-url.test.ts':
        '90db86e3b5217b89b18b082f37c502ef6750c54b2de58ad202fb71eecd9cfd29',
    'monitor-client.test.ts':
        '8dcff14f8fe5fb2f505ba7e9e845adab4f8c28c4af87c0234856817cecea4487',
    'panel-data.test.ts':
        'b3ece6c3f436f22b5abd3f7ffb295d3799f0fadf2628d4d791312bbd4445152a',
    'panel-view.test.ts':
        'bd2afe86800130e9ff8c03081a5696bd2c9f4c23fe71070c3fca57c72931d1b4',
    'mcp-bridge.test.ts':
        '6aceabcdaff4209df66f0380329bb6decded086e16051884823988698456c05c',
}


def _norm_sha(b):
    """行尾归一（CRLF→LF）后取 sha256（byte-exact 会被 core.autocrlf 打败）。"""
    return hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest()


def audit(table, read_fn):
    """纯函数：→ (问题列表, 统计)。喂合成 read_fn 即可证明它抓得住（R7）。"""
    problems, total_bytes = [], 0
    for name, want in sorted(table.items()):
        try:
            b = read_fn(name)
        except (OSError, KeyError) as e:      # 读不到（缺文件/路径不对）⇒ 记问题，不炸
            problems.append('%s：读不到（%s）' % (name, e.__class__.__name__))
            continue
        total_bytes += len(b)
        got = _norm_sha(b)
        if got != want:
            problems.append('%s：指纹不符（漂移）期望 %s… 实得 %s…'
                            % (name, want[:12], got[:12]))
    return problems, {'载体数': len(table), '哈希字节': total_bytes}


def _read_asset(name):
    with io.open(os.path.join(ROOT, ASSET_DIR, name), 'rb') as f:
        return f.read()


def _run_shared_tests(names):
    """`node --test` 跑共享单测 → (计数 dict, 原始输出尾部)。

    ⚠ node 缺失/调用失败 ⇒ 返回空计数（调用方判红）：**"无法判定" ≠ "判定通过"**。
    """
    files = [os.path.join(ROOT, ASSET_DIR, n) for n in names if n.endswith('.test.ts')]
    if not files:
        return {}, '（表里没有共享单测）'
    try:
        p = subprocess.run(['node', '--test'] + files, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=180,
                           cwd=os.path.join(ROOT, 'dsh'))
    except (OSError, subprocess.SubprocessError) as e:
        return {}, '（node 调用失败：%s）' % e.__class__.__name__
    out = (p.stdout or '') + (p.stderr or '')
    counts = {}
    for m in re.finditer(r'^\s*(?:#|\u2139)\s*(tests|pass|fail|skipped|cancelled)\s+(\d+)',
                         out, re.M):
        counts[m.group(1)] = int(m.group(2))
    return counts, '\n'.join(out.splitlines()[-6:])


def main():
    problems, stats = audit(CARRIERS, _read_asset)
    print('[dsh_panel_asset] 出处：%s' % REFERENCE_REVISION)
    C.group('A 行为载体与参考实现逐字一致（行尾归一 sha256 == 常量；按内容读、不经 git）',
            [('无漂移（首个问题：%s）' % (problems[0][:90] if problems else '共 0 条'),
              not problems)])

    # ---- B 致死性：合成内容喂给同一个 audit（R7）----
    good = b'export const x = 1\n'
    want = _norm_sha(good)
    cases = [
        ('干净内容 → 零问题', {'a.ts': want}, {'a.ts': good}, True),
        ('改一个字符 → 报出', {'a.ts': want}, {'a.ts': good + b' '}, False),
        ('删掉一个字符 → 报出', {'a.ts': want}, {'a.ts': b'export const x = '}, False),
        ('文件读不到 → 报出', {'missing.ts': want}, {}, False),
    ]
    subs = []
    for label, table, files, want_clean in cases:
        got, _st = audit(table, lambda n, f=files: f[n])
        subs.append(('%s（问题 %d 条）' % (label, len(got)), (not got) == want_clean))
    C.group('B 致死性：改一个字符 / 少一个文件必须报（非哑守卫）', subs)

    # ---- C 共享单测真跑（skipped 必须为 0）----
    counts, tail = _run_shared_tests(list(CARRIERS))
    stats['共享单测'] = counts
    C.group('C 共享单测真跑：pass>0 且 skipped==0（"全被跳过"是红，不是绿）', [
        ('node --test → %s' % (counts or '（没解析到计数）'),
         bool(counts) and counts.get('pass', 0) > 0 and counts.get('skipped', 0) == 0
         and counts.get('fail', 0) == 0),
        ('R8 自证：跑出的用例数 > 0（%d）' % counts.get('tests', 0),
         counts.get('tests', 0) > 0),
    ])
    if counts.get('fail') or counts.get('skipped') or not counts:
        print('[dsh_panel_asset] node 输出尾部：\n%s' % tail)

    print('[dsh_panel_asset] 非退化自证：载体 %d 个 / 哈希 %d 字节 / %s'
          % (stats['载体数'], stats['哈希字节'], stats['共享单测']))
    C.group('R8 非退化自证：钉了载体、真哈希了字节', [
        ('载体 %d 个 > 0、哈希字节 %d > 0' % (stats['载体数'], stats['哈希字节']),
         stats['载体数'] > 0 and stats['哈希字节'] > 0),
    ])

    ok = C.summary()
    for p in problems:
        print('   - %s' % p)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
