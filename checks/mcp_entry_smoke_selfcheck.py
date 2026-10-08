# -*- coding: utf-8 -*-
"""`--mcp` 入口**真子进程冒烟**：能起、端口文件**新鲜**、端点**真在听**、收尾干净。

## 它守的承诺（结构断言守不到的那半）

`checks/site_declaration_selfcheck.py` 只证明"声明写得对"；**这条**证明"照声明起，真能起来"：
起一个真 `app_entry.py --mcp`，等到**端口文件相对 spawn 前发生变化**（框架就绪判据的第一条），
再对**回读**到的端口打一次 `GET /mcp`，断言**拿到 HTTP 响应**（实测 406 ⇒ "有响应即在听"），
最后 terminate 并**等它真退出**（写租约随之释放，这是换腿/收尾的时序前提）。

⚠ **能红条件**（R7，确定性、不靠概率）：把 argv 的 `--mcp` 改成不存在的形态
（例如旧名 `--authority`）⇒ 子进程当场退出、端口文件永不变化 ⇒ 本判据红。

⚠ **不进 `--core/--fast`**：它起真服务（ARC cache + assemble，秒级起不来）。
⚠ 数据根用**临时目录**（`RE0_V2_DATA_ROOT`），**不碰仓内 `.mecha-v2`**；
端口文件是被测对象（项目根的运行时产物），跑完清掉。
"""
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402

C = Checker('mcp_entry_smoke')
ROOT = pathlib.Path(__file__).resolve().parent.parent
PF = ROOT / '.mcp-port'
MPF = ROOT / '.monitor-port'
TIMEOUT_S = 120.0


def _read(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding='utf-8').strip()
    except OSError:
        return ''


def _probe(url: str) -> str:
    """→ `'HTTP <code>'`（**任何** HTTP 响应都算在听）/ `'ERR <类>'`。"""
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return 'HTTP %d' % r.status
    except urllib.error.HTTPError as e:
        return 'HTTP %d' % e.code
    except Exception as e:                                    # noqa: BLE001
        return 'ERR %s' % type(e).__name__


def main() -> int:
    from mecha.launch import load_sites
    os.environ['MECHA_SITE_ROOTS'] = str(ROOT)
    site = next(s for s in load_sites() if any(v.id == 'EL' for v in s.services))
    svc = next(v for v in site.services if v.id == 'EL')

    tmp = tempfile.mkdtemp(prefix='el_mcp_smoke_')
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in svc.env.items()})
    env['RE0_V2_DATA_ROOT'] = tmp            # ← 覆盖成临时根：不碰仓内 journal
    before_pf, before_mpf = _read(PF), _read(MPF)
    log = open(os.path.join(tmp, 'child.log'), 'w', encoding='utf-8', errors='replace')
    proc = subprocess.Popen(list(svc.argv), cwd=str(svc.cwd), env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    port, code, mpf_after = None, 'ERR Timeout', ''
    try:
        deadline = time.time() + TIMEOUT_S
        while time.time() < deadline:
            time.sleep(0.25)
            if proc.poll() is not None:
                break
            cur = _read(PF)
            if cur and cur != before_pf and cur.isdigit():
                port = int(cur)
                code = _probe('http://127.0.0.1:%d/mcp' % port)
                mpf_after = _read(MPF)
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        tail = open(os.path.join(tmp, 'child.log'), encoding='utf-8',
                    errors='replace').read()[-400:]
        for f in (PF, MPF):                  # 端口文件是运行时产物：跑完清掉
            try:
                f.unlink()
            except OSError:
                pass

    C.group('真子进程冒烟：--mcp 能起 / 端口文件新鲜 / 端点真在听 / 收尾干净', [
        ('子进程起得来且**没当场退出**（存活到探活那一刻）', port is not None),
        ('端口文件相对 spawn 前**发生了变化**（%r → %r）'
         % (before_pf, str(port) if port else '未变'),
         port is not None and str(port) != before_pf),
        ('GET http://127.0.0.1:<port>/mcp → %s（任何 HTTP 响应都算在听）' % code,
         code.startswith('HTTP ')),
        ('只读监控端口文件也跟着换了（%r → %r）' % (before_mpf, mpf_after or '未变'),
         bool(mpf_after) and mpf_after != before_mpf),
        ('terminate 后真退出（写租约随之释放——换腿/收尾的时序前提）',
         proc.poll() is not None),
    ])
    ok = C.summary()
    if not ok:
        print('[mcp_entry_smoke] 子进程日志尾部：\n%s' % tail)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
