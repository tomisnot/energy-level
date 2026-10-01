# -*- coding: utf-8 -*-
"""dsh 面板**共享资产**守卫（EL 侧）：**只剩项目值**——指纹表 / 载体清单 / 用例下限 / 出处。

规则件已上提框架 `mecha.asset_check`（`7d512bc`）：正文指纹归一、逐件比对、
`node --test` 四判（退出码 / `fail==0` / `skipped==0` / `pass>=下限`）。
⇒ 本项目**不再自己实现** sha 归一、node 调用与计数解析（此前两个项目各写了一份 218 行、
形态还不同）。

## 承诺没变（只是换了持有者）
- **A 逐件比对全部载体** ⇒ `audit_carriers`（缺件/读不到都是红，且读得出是哪个文件）；
- **C 共享单测真跑** ⇒ `run_node_suite`：**硬依赖 `node`（缺即红，不软跳过）**，
  且 **`skipped != 0` 判红**（"跑起来了但没有用例真的跑"是最坏的一种绿）。
- **B 致死性（R7）留在这里**：把合成表/合成 `read_fn` 喂给**同一个** `audit_carriers`，
  证明它抓得住"改一个字符 / 少一个文件"。⚠ 这条测的是**我用对了这个库**（表与路径接对没），
  不是重测库本身——库自己的守卫在 `mecha/checks/asset_check_selfcheck.py`。
- **R8 非退化自证**：打印载体数 / 哈希字节 / 四计数（否则"表是空的"也会看起来通过）。

## 复制约定（不变的纪律）
`dsh/src/dsh-panel/` 是 `mecha/dsh-panel/` 的逐字副本：**只有参数块 `panel-config.ts`
允许项目化**，其余文件必须逐字一致。出处见 `REFERENCE_REVISION`——**别在别处另抄一个
提交哈希**（D1：事实一个家）。

用法：python checks\\dsh_panel_asset_selfcheck.py     退出码 0 全过 / 1 有问题。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402
from mecha.asset_check import (                              # noqa: E402
    audit_carriers, run_node_suite, text_sha256,
)

C = Checker('dsh_panel_asset')
ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / 'dsh' / 'src' / 'dsh-panel'

#: **出处说明**（只作说明，不参与比对——比对靠下面那组常量）。
REFERENCE_REVISION = ('mecha/dsh-panel @ 22e41cc（2026-10-01：桥修 `ensureReady` 移入 `try` —— '
                      '补回移植时丢掉的那处修复，"首连失败后再调用"要回可读离线错误；'
                      '并只对连接/会话类失败包装"服务离线"（编程错不再被伪装）；'
                      '更早 c0cd1d3〔opt-in 件首批〕、7490db2、58e2f8a、f516a99、c44b01f）')

#: **行为载体**（代码 + 单测）→ 行尾归一 sha256。改这里 = "上游发新版后整份重新复制"。
#: ⚠ `panel-config.ts` 不进表（它是参数块，必然不同）；`README.md` 不进表（prose）。
#: 2026-10-01 起本目录含两类标准件（资产 README 的分类）：**默认件**（面板 + 地址路由）
#: 与 **opt-in 件**（自愈 MCP 桥 + 工具注册 + SDK 类型垫片）；EL 两类都用。
#: ⚠ opt-in 件的**项目值经参数传进去**（`dsh/src/el-params.ts` / `host/el-hub.ts`，
#: 那两个文件不是资产件、不进本表）⇒ 本表逐字一致 = 合同；参数填错由 EL 自己的单测抓。
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
    # opt-in 件：自愈 MCP 桥 + 端点解析 + 工具注册 + SDK 类型垫片
    'mcp-bridge.ts':
        '59c5078fce46c4610a319d2a70628ef7827ea11c01734be02eca67d35ca2d510',
    'mcp-session-http.ts':
        '423dca6f422491690e91c03b1576296b578162aa5be72684419bc510445fa194',
    'config.ts':
        '537668e2c0b4af8bc0b624ec42cd9e5c0f126205ab676e7700bc592e137c4207',
    'register-tools.ts':
        '4eadbf2f495f2f16e09a197db920b1662cb17ddbf051f563509fbe08e0174bf0',
    'mcp-sdk-shims.d.ts':
        'd2fc24f0d7131fa2c2e700e9d669c51578f0698023c7919b3b3d36f027aa331e',
    # 单测（随实现一起复制；由 C 组真跑——清单从表里来，不手抄，D4）
    'monitor-url.test.ts':
        '90db86e3b5217b89b18b082f37c502ef6750c54b2de58ad202fb71eecd9cfd29',
    'monitor-client.test.ts':
        '8dcff14f8fe5fb2f505ba7e9e845adab4f8c28c4af87c0234856817cecea4487',
    'panel-data.test.ts':
        'b3ece6c3f436f22b5abd3f7ffb295d3799f0fadf2628d4d791312bbd4445152a',
    'panel-view.test.ts':
        'bd2afe86800130e9ff8c03081a5696bd2c9f4c23fe71070c3fca57c72931d1b4',
    'mcp-bridge.test.ts':
        '4fba5f5e963da321a1b3d33a4fc67323e0506cfad18d2fdadf7df7cc7962dc7e',
}

#: 要跑的共享单测 = 表里的测试件（**单一来源**，不另抄清单）。
SHARED_TESTS = tuple(n for n in CARRIERS if n.endswith('.test.ts'))

#: 用例数**下限**（防塌缩，不是精确计数）：实际条数以输出里打印的为准。
#: ⚠ 不许大于当前真实用例数——否则资产一加用例，判据反而红。
MIN_PASS = 40


def _read(name: str) -> bytes:
    return (ASSET_DIR / name).read_bytes()


def _read_raises(table) -> bool:
    """`audit_carriers` 在 read 抛错时**必须**让错冒出来（缺件不许当"符"）。"""
    def _boom(_n):
        raise FileNotFoundError('synthetic missing')
    try:
        audit_carriers(table, _boom)
    except FileNotFoundError:
        return True
    return False


def main() -> int:
    print('[dsh_panel_asset] 出处：%s' % REFERENCE_REVISION)
    print('[dsh_panel_asset] 资产目录：%s（载体 %d 个）' % (ASSET_DIR, len(CARRIERS)))

    # ---- A 逐件比对（规则在 mecha.asset_check；本仓只给表）----
    try:
        drifted = audit_carriers(CARRIERS, _read)
        total_bytes = sum(len(_read(n)) for n in CARRIERS)
    except OSError as e:                    # 缺件/读不到：红，且读得出是哪个文件
        drifted, total_bytes = ['载体读取失败：%s' % e], 0
    C.group('A 行为载体与参考实现逐字一致（正文指纹；缺件/漂移都算红）',
            [('无漂移（首个问题：%s）' % (drifted[0][:90] if drifted else '共 0 条'),
              not drifted)])

    # ---- B 致死性：合成内容喂给**同一个** audit_carriers（R7 测"我接对了表"）----
    good = b'export const x = 1\n'
    want = text_sha256(good)
    cases = [
        ('干净内容 → 零问题', {'a.ts': want}, {'a.ts': good}, True),
        ('改一个字符 → 报出', {'a.ts': want}, {'a.ts': good + b' '}, False),
        ('删掉一个字符 → 报出', {'a.ts': want}, {'a.ts': b'export const x = '}, False),
    ]
    subs = []
    for label, table, files, want_clean in cases:
        got = audit_carriers(table, lambda n, f=files: f[n])
        subs.append(('%s（问题 %d 条）' % (label, len(got)), (not got) == want_clean))
    subs.append(('文件读不到 → 抛（不静默当"符"）',
                 _read_raises({'missing.ts': want})))
    C.group('B 致死性：改一个字符 / 缺一个文件必须报（非哑守卫）', subs)

    # ---- C 共享单测真跑（规则在库：四判 + 缺 node 即红）----
    r = run_node_suite(ASSET_DIR, SHARED_TESTS, min_pass=MIN_PASS)
    C.group('C 共享单测真跑：退出码 0 / fail==0 / skipped==0 / pass>=%d' % MIN_PASS,
            [('%s —— %s' % (r.summary, r.detail), r.ok)])

    # ---- R8 非退化自证 ----
    C.group('R8 非退化自证：钉了载体、真哈希了字节、真跑了用例',
            [('载体 %d 个 > 0、哈希 %d 字节 > 0、实跑 %d 条 > 0'
              % (len(CARRIERS), total_bytes, r.counts.get('tests', 0)),
              len(CARRIERS) > 0 and total_bytes > 0 and r.counts.get('tests', 0) > 0)])

    ok = C.summary()
    for p in drifted:
        print('   - %s' % p)
    if not r.ok:
        print('[dsh_panel_asset] node 输出尾部：\n%s'
              % '\n'.join(r.output.splitlines()[-6:]))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
