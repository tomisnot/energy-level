# -*- coding: utf-8 -*-
"""自检：`.mcp-port` 的**归属权**——没收养 server 的进程不得删别人的端口文件。

事故（2026-09-12 实测）：主代理的软件实例正在 127.0.0.1:8765 上跑实验，
AI 自检批次里 `checks/ui_selftest.py` 调了 `win.close()` → `MainWindow.closeEvent`
→ `ai_server.stop_server()` → 原来的实现**无条件** `os.remove('.mcp-port')`，
把活实例的端口文件删掉了（客户端按文件发现端口 → 直接失败）。

本自检覆盖修好后的两道判据：
  A 本进程从没收养 server（自检/GUI 检查的常态）→ **不碰**端口文件；
  B 收养了且文件内容 == 本进程端口 → 删（正常退出路径行为不变）；
  C 收养了但文件属于**别的**端口（重启竞态）→ 不删；
  D 文件不存在 / srv 未就绪 → 不抛异常。

注意：本自检把 `server._PORT_FILE` 指向临时文件，**绝不碰**项目根真实
`.mcp-port`（可能正被活实例使用）——跑完恢复原值。
用法：python checks/port_file_ownership_selfcheck.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402
from src.ai import server as S                                   # noqa: E402

C = Checker('port_file_ownership')

_REAL = S._PORT_FILE
_tmp = os.path.join(tempfile.gettempdir(), '_el_port_ownership_probe')
S._PORT_FILE = _tmp


class _Srv:
    should_exit = False


def _write(txt):
    with open(_tmp, 'w', encoding='utf-8') as f:
        f.write(txt)


def _exists():
    return os.path.exists(_tmp)


try:
    # A. 没收养 server（= 所有 GUI 自检进程的形态）
    S._server_state.clear()
    _write('8765')
    S.stop_server()
    C.check('A 没收养 server → 不删别人的端口文件', _exists())

    # B. 收养了、且文件就是自己写的那个端口 → 删（正常退出不变）
    S._server_state.clear()
    S._server_state['srv'] = _Srv()
    S._server_state['port'] = 8765
    _write('8765')
    S.stop_server()
    C.check('B 自己的端口文件 → 正常删除', not _exists())
    C.check('B2 srv.should_exit 已置位（真的停 server）',
            S._server_state['srv'].should_exit is True)

    # C. 收养了，但文件属于别的实例（重启竞态）
    S._server_state.clear()
    S._server_state['srv'] = _Srv()
    S._server_state['port'] = 8765
    _write('41000')
    S.stop_server()
    C.check('C 文件属于别的端口（新实例刚写的）→ 不删', _exists())

    # D. 文件不存在 / 端口未知 → 不抛
    S._server_state.clear()
    try:
        os.remove(_tmp)
    except OSError:
        pass
    S.stop_server()
    C.check('D 文件不存在 → 不抛异常', True)

    S._server_state.clear()
    S._server_state['srv'] = _Srv()          # 有 srv 但 port 未知（异常路径）
    _write('8765')
    S.stop_server()
    C.check('D2 port 未知 → 保守不删（宁可留陈旧文件也不删活文件）', _exists())
finally:
    S._PORT_FILE = _REAL
    S._server_state.clear()
    try:
        os.remove(_tmp)
    except OSError:
        pass

ok = C.summary()
sys.exit(0 if ok else 1)
