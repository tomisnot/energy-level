# -*- coding: utf-8 -*-
"""Rb-87 看门人（launcher）：两模式启动与交接的**唯一编排者**。

心智模型（用户裁决 2026-09-24，两模式收拢）：
- **GUI 模式**：纯本地软件（`app_entry.py --gui`），写权在人，AI 物理看不见
  （不起 MCP/监控端点）。想动 AI 就切到 AI 模式。
- **AI 模式**：headless 权威（`app_entry.py --authority`，写权自动在 AI）+ dsh
  前台（AI 界面 + 驾驶室/监控面板）。人在 dsh 里点"切换至 GUI"就切回。

看门人常驻（无窗口，日志进本控制台），职责：
1. 启动弹窗选一次模式 → spawn 对应进程组。
2. 轮询 `.mode-request`（任一子进程写它 = 切换意图）→ 交接：terminate 当前组、
   **等进程真退出**（journal 写租约随进程释放，否则新权威被 writer_lease_held 拒）、
   按新模式 spawn、删请求文件。
3. 主进程自然退出且无切换请求 → 收尾退出。

**写权落位**：GUI 模式 make_v2_bridge 出厂即 HUMAN；AI 模式 run_authority 起
权威后 switch_mode(AI, actor='human')——看门人**代理人的启动意图**开闸（人选了
AI 模式即授权 AI 写），不是 AI 自解锁（护栏仍在：AI 面改不了 mode）。语义见
mecha 仓 ADR `两模式启动-看门人交接与写权自动落位`。

用法：python launcher.py   （或双击 能级.py 转发到这里）
"""
import os
import shutil
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
APP = os.path.join(HERE, "app_entry.py")

MODE_REQUEST = os.path.join(HERE, ".mode-request")   # 子进程写：切换意图（gui/ai）
MODE_STATE = os.path.join(HERE, ".mode-state")       # 看门人写：当前模式（供人/判据看）
MCP_PORT = os.path.join(HERE, ".mcp-port")           # 权威写：dsh 插件据此发现端点

GUI, AI = "gui", "ai"
_FLAGS = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0


def _log(msg):
    print("[launcher] %s" % msg, flush=True)


# ---------------- 进程组：一个模式 = 一组子进程 ----------------

def _spawn(cmd):
    """起一个子进程，输出继承本控制台（dsh 启动进度/URL 可见）。"""
    return subprocess.Popen(cmd, cwd=HERE, creationflags=_FLAGS)


def start_group(mode):
    """按模式 spawn 进程组，返回 (procs, primary)。primary=主进程：它自然退出
    且无切换请求 ⇒ 看门人收尾。GUI 模式 primary=GUI 窗口；AI 模式 primary=dsh。"""
    if mode == GUI:
        return [_spawn([PY, APP, "--gui"])], None   # primary 用索引 0
    # AI 模式：先起 headless 权威，等它写 .mcp-port，再起 dsh（前台）
    auth = _spawn([PY, APP, "--authority"])
    _wait_mcp_port()
    dsh = _spawn_dsh()
    return [auth, dsh], 1                            # primary=dsh（索引 1）


def _wait_mcp_port(timeout=90):
    """等权威写好 .mcp-port（cache 建 + 装配要十几秒）。超时不致命——dsh 退回默认端口。"""
    for _ in range(int(timeout / 0.25)):
        try:
            if os.path.exists(MCP_PORT) and open(MCP_PORT, encoding="utf-8").read().strip():
                return True
        except OSError:
            pass
        time.sleep(0.25)
    _log("⚠ 等 .mcp-port 超时——dsh 会退回默认 :8765（权威可能没起来）")
    return False


def _pick_dsh_port(preferred=3081, tries=10):
    """从 preferred 起找第一个可绑定回环端口；**3080 硬跳过**（官方 dsh 的端口，
    项目实例永不占——注释即机制，不靠默认值兜底）。"""
    for p in range(preferred, preferred + tries):
        if p == 3080:
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
    return preferred


def _spawn_dsh():
    dsh_exe = shutil.which("dsh")
    dsh_dir = os.path.join(HERE, "dsh")
    if not dsh_exe:
        _log("⚠ PATH 里没找到 dsh——AI 模式只有权威在跑，无 AI 界面。"
             "手动运行：dsh web --patch ./dsh/cordis.source.patch.yml "
             "--patch ./dsh/cordis.project.patch.yml")
        return None
    # 依赖预检：node_modules 缺失 → 自动 npm install（否则插件 import MCP SDK 失败崩）
    if os.path.isdir(dsh_dir) and not os.path.isdir(os.path.join(dsh_dir, "node_modules")):
        _log("dsh/node_modules 缺失 → 自动 npm install（首次约 15-60s）…")
        try:
            subprocess.run("npm install", shell=True, cwd=dsh_dir)
        except Exception as e:                          # noqa: BLE001
            _log("⚠ npm install 失败：%s（dsh 可能起不来）" % e)
    port = _pick_dsh_port()
    args = "web --patch ./dsh/cordis.source.patch.yml " \
           "--patch ./dsh/cordis.project.patch.yml --port %d" % port
    _log("起 dsh（本实例 :%d，避开官方 :3080）；起来后自动开浏览器 = AI 界面" % port)
    return _spawn('"%s" %s' % (dsh_exe, args))


# ---------------- 交接：terminate 当前组 → 等退出（租约释放）→ spawn 新模式 ----------------

def _terminate_group(procs):
    for p in procs:
        if p is None:
            continue
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:                               # noqa: BLE001
            pass


def _await_exit(procs, timeout=15):
    """等进程真退出（写租约随进程释放）。超时才 kill——防交接窗口内租约没放干净。"""
    deadline = time.time() + timeout
    for p in procs:
        if p is None:
            continue
        while p.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if p.poll() is None:
            try:
                p.kill()
            except Exception:                           # noqa: BLE001
                pass


def _read_request():
    try:
        if os.path.exists(MODE_REQUEST):
            v = open(MODE_REQUEST, encoding="utf-8").read().strip().lower()
            return v if v in (GUI, AI) else None
    except OSError:
        pass
    return None


def _clear_request():
    try:
        if os.path.exists(MODE_REQUEST):
            os.unlink(MODE_REQUEST)
    except OSError:
        pass


def _write_state(mode):
    try:
        open(MODE_STATE, "w", encoding="utf-8").write(mode)
    except OSError:
        pass


# ---------------- 主循环 ----------------

def _choose_mode():
    from PyQt5.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication(sys.argv)
    box = QMessageBox()
    box.setWindowTitle("Rb-87 能级计算 · 选择启动模式")
    box.setIcon(QMessageBox.Question)
    box.setText("请选择本次启动的模式：")
    box.setInformativeText(
        "AI 模式：软件在后台当权威（写权自动交给 AI），前台起 dsh——你在 dsh 里\n"
        "用 AI 驱动引擎，右侧栏监控它的每笔操作；想收回控制权点 dsh 里的「切换至 GUI」。\n\n"
        "GUI 模式：纯本地软件，写权在你手上，AI 完全看不见这台软件；\n"
        "想用 AI 就在菜单里点「切换至 AI 模式」。\n\n"
        "两种模式共用同一本账（journal），切换时状态原样交接、不断档。")
    b_ai = box.addButton("AI 模式", QMessageBox.AcceptRole)
    b_gui = box.addButton("GUI 模式", QMessageBox.ActionRole)
    box.addButton("退出", QMessageBox.RejectRole)
    box.setDefaultButton(b_gui)
    box.exec_()
    c = box.clickedButton()
    if c is b_ai:
        return AI
    if c is b_gui:
        return GUI
    return None


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _clear_request()
    mode = _choose_mode()
    if mode is None:
        _log("未选模式，退出。")
        return 0
    _log("启动 %s 模式。" % ("GUI（人在开）" if mode == GUI else "AI（AI 在开）"))
    procs, primary = start_group(mode)
    _write_state(mode)

    try:
        while True:
            time.sleep(0.5)
            req = _read_request()
            if req and req != mode:
                _log("收到切换请求：%s → %s，交接中…" % (mode, req))
                _terminate_group(procs)
                _await_exit(procs)
                _clear_request()
                procs, primary = start_group(req)
                mode = req
                _write_state(mode)
                _log("已切到 %s 模式。" % mode)
                continue
            # 无切换请求：主进程自然退出 ⇒ 收尾
            prim = procs[primary] if primary is not None else procs[0]
            if prim is not None and prim.poll() is not None:
                _log("主进程已退出（%s 模式结束），收尾。" % mode)
                break
    except KeyboardInterrupt:
        _log("\nCtrl+C —— 收尾所有子进程。")
    finally:
        _terminate_group(procs)
        _await_exit(procs, timeout=8)
        _clear_request()
        try:
            if os.path.exists(MODE_STATE):
                os.unlink(MODE_STATE)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
