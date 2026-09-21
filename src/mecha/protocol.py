# -*- coding: utf-8 -*-
"""引擎协议契约（框架侧声明）——`mecha` 只认这个接口，**不认任何领域**。

"拷走 `mecha/`+`cockpit/` 就等于提炼走框架"的关键：mecha 全程只依赖本 ABC 声明的方法，
领域实现（本项目 = `src/adapter.EngineAdapter` 包装 `src/core/protocol`）在 app 组装时
经 `mecha.runtime.set_engine(...)` **注入**。新项目接入本框架，只需实现一个
`EngineProtocol`，mecha 一行不改。

**依赖方向**：`adapter → mecha.protocol + core`；`mecha ↛ core/adapter/arc`（提炼守卫强制）。
"""
from abc import ABC, abstractmethod

__all__ = ['EngineProtocol']


class EngineProtocol(ABC):
    """mecha 对一个"引擎"要求的全部面。入参/出参都是普通 dict / 不透明对象——
    mecha 不解析其内部结构（那是领域的事），只搬运与编排（事件/产物/authority/缓存）。"""

    # ---------- schema（registry 收集；engine.* 只读） ----------
    @abstractmethod
    def engine_params(self):
        """→ {name: record}：引擎就地常量元数据（family='engine', readonly=True）。"""

    @abstractmethod
    def config_schema(self):
        """→ {key: record}：build_run 读的配置键 schema。"""

    @abstractmethod
    def solve_schema(self):
        """→ {'solve.X': record}：运行级旋钮（含 engine_kwarg；可写）。"""

    @abstractmethod
    def solve_map(self):
        """→ {'MAX_STEP': 'max_step', ...}：solve.* → 引擎 kwarg（唯一映射）。"""

    @abstractmethod
    def settable_keys(self):
        """→ [key]：可写键全集 = config + solve（**不含 engine.***）。"""

    @abstractmethod
    def validate_wave_list(self, value):
        """校验 serialize_waves 同构 list；非法抛 mecha 的 InvalidValue（适配器注入错误类型）。"""

    # ---------- cache（不透明句柄；mecha 不认识 arc） ----------
    @abstractmethod
    def make_cache(self):
        """→ 引擎缓存句柄（本项目 = ArcCache）。mecha 只透传。"""

    # ---------- run（唯一分派；mecha 只编排事件/产物/authority） ----------
    @abstractmethod
    def build_run(self, cache, snapshot, diag=False):
        """配置快照 → RunSpec（不透明）。"""

    @abstractmethod
    def check_run(self, run):
        """→ [问题字符串]：运行前自检。"""

    @abstractmethod
    def evolve_run(self, run, progress_arr=None):
        """RunSpec → 结果 dict（不透明；含 pops/states/…）。"""

    @abstractmethod
    def result_summary(self, run, res, **kw):
        """结果 → 紧凑摘要（不搬大数组）。"""

    @abstractmethod
    def summarize_run(self, run):
        """RunSpec → 人类/AI 可读摘要。"""

    # ---------- query（纯查询，不跑演化） ----------
    @abstractmethod
    def estimate(self, cache, snapshot):
        """成本预估（dry-run）。"""

    @abstractmethod
    def preview(self, cache, snapshot):
        """行为预览（驱动哪些跃迁、基组够不够、有无重叠）。"""

    @abstractmethod
    def list_levels(self, cache, snapshot):
        """当前基组的态（label + 身份键 + 能量）——语义寻址，不猜索引。"""

    @abstractmethod
    def run_annotations(self, snapshot):
        """运行旁注：{n_waves, overlap_warning}（供 run 的 provenance）。"""

    # ---------- 可选领域能力（建场层等） ----------
    def make_fields(self, session):
        """→ 建场层对象（领域可选能力）；默认无。mecha 只透传其公开方法，不认识具体动词。"""
        raise NotImplementedError('本引擎未提供建场层')
