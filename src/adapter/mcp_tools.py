# -*- coding: utf-8 -*-
"""本领域的 MCP 工具（**从 mecha 库搬回来的**，P1-9d）。

为什么它们不该在库里：`set_drive`/`get_waves` 的签名里全是物理量
（`rabi_mhz` / `intensity_mwcm2` / `detuning_mhz` / `pol='π'` / `pulse` / `edge_ns`），
库把它们当一等工具 ⇒ 一个文献/数据流水线项目接进来会**凭空多出两个物理工具**
（控诉书罪一/L1）。库现在只提供**扩展点**（`engine.mcp_tools()`），
"注册什么"由本文件给——**领域知识留在领域侧**。

工具对象只需三个属性（库只按这三样注册，不解释其内容）：
  · `fn`          ：可调用，**签名即 MCP 输入 schema**（返回 JSON 字符串）；
                    若 `needs_fields=True`，库会以 `fields=<建场层>` 关键字注入
  · `description` ：工具描述
  · `needs_fields`：True = 需要建场层；无建场层时库仍注册但调用回**可教学错误**

⚠ **不要在本模块用 `from __future__ import annotations`**：那会把注解变成字符串，
而库用 `typing.get_type_hints()` 把它们解析成**真实类型对象**去构造 MCP 输入 schema；
字符串注解在库的命名空间里解析不到 `Optional` ⇒ pydantic 报错。
"""
from typing import Any, Optional

__all__ = ['build_tools']


class DomainTool:
    """最小工具描述符（库只读 fn/description/needs_fields 三个属性）。"""

    __slots__ = ('fn', 'description', 'needs_fields')

    def __init__(self, fn, description, needs_fields=True):
        self.fn = fn
        self.description = description
        self.needs_fields = needs_fields


def set_drive(*, fields, reason: str, src: Any, tgt: Any, atom: str = 'single',
              pol: str = 'π', rabi_mhz: Optional[float] = None,
              intensity_mwcm2: Optional[float] = None,
              detuning_mhz: float = 0.0, pulse: str = '',
              dur_ns: Optional[float] = None, edge_ns: float = 1.0,
              name: str = '') -> str:
    """L2 意图级建场：在某跃迁上建驱动。

    给 src/tgt（label 或身份键）+ Rabi 或强度 + 脉冲类型(pi/pi2)或时长
    → 自动算 freq/intensity/duration。`fields` 由库注入（无建场层时不会调到本函数）。
    """
    return fields.set_drive(atom, src, tgt, pol=pol, rabi_mhz=rabi_mhz,
                            intensity_mwcm2=intensity_mwcm2,
                            detuning_mhz=detuning_mhz, pulse=(pulse or None),
                            dur_ns=dur_ns, edge_ns=edge_ns,
                            name=(name or None), actor='ai', reason=reason)


def get_waves(*, fields, atom: str = 'single') -> str:
    """读回某原子的波形摘要（不内联大数组；要曲线走 CodeAct 的 get_envelope）。"""
    return fields.get_waves(atom)


def build_tools():
    """→ `{name: DomainTool}`：本领域自报给 `create_server` 的额外工具。

    库用 `inspect.signature(fn)` 推 MCP 输入 schema，并**跳过 `fields` 参数**
    （那是库注入的，不该出现在 AI 看到的 schema 里）。
    """
    return {
        'set_drive': DomainTool(
            set_drive,
            'L2 意图级建场：在某跃迁上建驱动。给 src/tgt（label 或身份键）+ '
            'Rabi 或强度 + 脉冲类型(pi/pi2)或时长 → 自动算 freq/intensity/duration。',
            needs_fields=True),
        'get_waves': DomainTool(
            get_waves,
            '读回某原子的波形摘要（不内联大数组；要曲线走 CodeAct get_envelope）。',
            needs_fields=True),
    }
