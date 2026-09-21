# -*- coding: utf-8 -*-
"""可教学错误（纲领 D2：错误消息是 LLM 的 UI）。

打错键名时**先给最可能的那个**（含别名/相似项线索），再给"要全量就调列表"。
糊弄的空间藏在"AI 声称的"与"实际发生的"之间的缝里；一条**能自己教 AI 改对**
的错误消息，是把这条缝收窄的最高 ROI 手段（`old/AI-调用架构.md` §4 第 5 条：
"失败即修法"）。

设计要点：
- `MechaError` 携带结构化字段 `kind/hint/suggest`（不只是散文），便于 MCP 回程
  与驾驶舱告警消费（harvest：构造性错误 kind/hint/suggest）。
- `did_you_mean` 用 `difflib` 给相似项——**别名要么是"人真会打的写法"，要么空着**
  （纲领 D2：一半是噪声的字段等于没有）。
"""
import difflib

__all__ = ['MechaError', 'UnknownKey', 'InvalidValue', 'ContractError',
           'did_you_mean']


class MechaError(Exception):
    """机甲框架的结构化错误基类。

    字段：
      kind    : 机器可读的错误类别（'unknown_key' / 'invalid_value' / ...）
      hint    : 一句"下一步该怎么办"的可教学提示
      suggest : 候选纠正列表（如相似键名），可为空
    """
    kind = 'mecha_error'

    def __init__(self, message, hint=None, suggest=None, **extra):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.suggest = list(suggest or [])
        self.extra = extra

    def to_dict(self):
        """→ 可 JSON 序列化的错误体（MCP 回程 / 事件总线用）。"""
        d = {'kind': self.kind, 'message': self.message}
        if self.hint:
            d['hint'] = self.hint
        if self.suggest:
            d['suggest'] = self.suggest
        if self.extra:
            d.update(self.extra)
        return d

    def __str__(self):
        parts = [self.message]
        if self.suggest:
            parts.append('你是不是想写：%s' % '、'.join(self.suggest))
        if self.hint:
            parts.append(self.hint)
        return '  '.join(parts)


class UnknownKey(MechaError):
    """键名不在契约内（I6：契约外字段读出来永远是 None，写进去必须报错）。"""
    kind = 'unknown_key'


class InvalidValue(MechaError):
    """值违反类型/单位/值域/前置条件（P0.4：契约自由度锁死）。"""
    kind = 'invalid_value'


class ContractError(MechaError):
    """契约一致性问题（I4：安全口径/单位/时间戳形态不一致等）。"""
    kind = 'contract_error'


def did_you_mean(name, candidates, n=3, cutoff=0.6):
    """给 `name` 在 `candidates` 里的最可能纠正（difflib 相似度）。

    返回最多 n 个候选，按相似度降序。**只返回真像的**（cutoff 之上）——
    宁缺毋滥（D2：一半是噪声的建议等于没有）。
    """
    if not name or not candidates:
        return []
    matches = difflib.get_close_matches(str(name), [str(c) for c in candidates],
                                        n=n, cutoff=cutoff)
    return matches
