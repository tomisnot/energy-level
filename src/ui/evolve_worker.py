# -*- coding: utf-8 -*-
"""演化计算后台线程（独立于 UI，供动态页复用）。

阻塞式 evolve.evolve 放 QThread 后台，主线程轮询 progress_arr 更新进度/取消。
不依赖任何 UI 类；done/log_msg 信号由使用方连接。
"""
from PyQt5.QtCore import QThread, pyqtSignal

from ..core.evolve_util import decay, auto_decays  # t74 工具归位


class RunWorker(QThread):
    """演化后台线程（新架构·I5 人机同路径）：主线程已 `ops.prepare`（build_run 碰 cache），
    本线程只 `ops.execute`（evolve_run 纯数值、**不碰 cache**）。

    B-1① 定论：ARC 的 SQLite **线程亲和**（无 check_same_thread=False）→ 两体 build_run 的
    V_dd（大量 cache.dme）必须在**建 cache 的主线程**跑；只有 evolve_run 可进工作线程
    （见 `checks/ops_threadsplit_selfcheck`：execute 工作线程 cache 调用==0）。

    进度/取消沿用 `progress_arr` 协议（`[frac, cancel, stage]`；写 `[1]=1` 请求取消，
    引擎步间复核）。`done(object, object)` = (ops 结果 dict | None, error_str | None)；
    用户取消 → (None, 'cancelled')。结果的 pops/t 数组落 artifact（I3），GUI 侧用
    `ops.load_artifact(result['artifact_id'])` 取回绘图（人机同产物）。
    """

    done = pyqtSignal(object, object)       # (result_dict | None, error_str | None)
    log_msg = pyqtSignal(str)               # 运行旁注（problems 等）→ 状态栏

    def __init__(self, ops, prepared, progress_arr, *, actor='gui',
                 reason='GUI 运行', land_artifact=True, parent=None):
        super().__init__(parent)
        self.ops = ops
        self.prepared = prepared            # 主线程 ops.prepare 的 handle
        self.progress_arr = progress_arr
        self.actor = actor
        self.reason = reason
        self.land_artifact = land_artifact

    def run(self):
        try:
            res = self.ops.execute(self.prepared, actor=self.actor,
                                   reason=self.reason,
                                   progress_arr=self.progress_arr,
                                   land_artifact=self.land_artifact)
            for p in (res.get('problems') or []):
                self.log_msg.emit(str(p))
            self.done.emit(res, None)
        except RuntimeError as e:
            # 用户取消：progress_arr[1]≠0 → 不算错误，静默
            if self.progress_arr[1] != 0.0:
                self.done.emit(None, 'cancelled')
            else:
                self.done.emit(None, str(e))
        except Exception as e:                              # noqa: BLE001
            self.done.emit(None, str(e))


class HubRunWorker(QThread):
    """Hub 模式后台线程：run 在 **Hub 侧**（run_compute，auto async），本地不 prepare/execute。

    GUI 作 Hub 客户端（Phase 1）时的运行路径：单一真值源在 Hub，本地无 cache/RunSpec。
    结果 result dict 回传；artifact 在 Hub 侧，GUI 不 load 本地 artifact（hub 模式绘图
    降级为只报 summary）。done/log_msg 信号同 RunWorker。
    """

    done = pyqtSignal(object, object)
    log_msg = pyqtSignal(str)

    def __init__(self, bridge, reason='GUI 运行', parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.reason = reason

    def run(self):
        try:
            res = self.bridge.run(reason=self.reason)
            self.done.emit(res, None)
        except Exception as e:                              # noqa: BLE001
            self.done.emit(None, str(e))


