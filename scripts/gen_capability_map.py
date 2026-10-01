# -*- coding: utf-8 -*-
"""生成 EL 的 capability map（mecha 总纲 §5.4 交付 3：真实 EL 表）。

ADR `边界机制补三洞-运行时图与声明表一致`：并表判据不许再靠 mecha 侧手写的
合成表——**EL 自己生成一份真表**入库，mecha 判据读它做机械 diff。

表 = 装配点出厂声明（build_roles 注入 EL 的 engine_name/project）
   + EL 运行时真实注册项（build_tool_registry 往 Surface 登记的 QUERY 等）。
schema-only 引擎（照 checks/ai_surface_selfcheck 先例）：build 期只调 schema()，
不建 ARC cache，秒级。

用法：python scripts/gen_capability_map.py            # 生成 docs/capabilities.md
      python scripts/gen_capability_map.py --check    # 守卫：现文件与生成结果一致
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# mecha 仓根从已安装的 mecha 包反推（editable 安装 = 仓根的 mecha/）——
# 不硬编码绝对路径；mecha_v2 与 scripts 都在 EL 仓根下。
import mecha as _mecha  # noqa: E402
_MECHA_ROOT = Path(_mecha.__file__).resolve().parent.parent
if str(_MECHA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MECHA_ROOT))

from mecha.assembly import build_roles  # noqa: E402
from mecha.authority import Authority, Mode  # noqa: E402
from mecha.gate import Channel, Gate, accept_all  # noqa: E402
from mecha.history import History  # noqa: E402
from mecha.providers.jobs import JobRegistry  # noqa: E402
from mecha.surface import Surface  # noqa: E402

sys.path.insert(0, str(_MECHA_ROOT / "scripts"))
import gen_capability_map as _gen  # noqa: E402  （render 复用：字段格式单一来源）

OUT = _ROOT / "docs" / "capabilities.md"


def el_entries():
    """EL 真运行时图的角色条目（出厂声明 + 项目注册项）。"""
    from src.core import protocol as P
    from mecha_v2.tools import build_tool_registry

    class ElEngine:
        """map 生成期的 ElEngine 替身（只喂 schema()，不 run/estimate、不建 cache）。
        类名刻意与真引擎一致：Surface 自注册的 provider=type(engine).__name__
        要如实反映运行时真值（部署路径上它就是 ElEngine），map 不许出现替身名。"""

        def schema(self):
            return {"run_output_required": ["ok", "method"],
                    "estimate_output_required": ["est_sec"],
                    "config": P.config_schema(), "solve": P.solve_schema(),
                    "engine": P.engine_params(), "settable_keys": P.settable_keys()}

    eng = ElEngine()
    roles = build_roles(engine_name="ElEngine", project="energy-level")
    surface = Surface(eng, roles=roles)
    hist = History()
    auth = Authority(mode=Mode.AI)
    gate = Gate(auth, hist, validate=accept_all)
    # 真注册：QUERY 项等落进同一张 roles 表（与部署路径 build_tool_registry 同源）
    build_tool_registry(eng, gate, surface, auth, Channel("ai"), JobRegistry())
    return surface.roles.entries()


def render() -> str:
    return _gen.render(el_entries())


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    generated = render()
    if "--check" in sys.argv:
        if not OUT.exists():
            print("EL docs/capabilities.md 缺失，请运行本脚本生成")
            return 1
        if OUT.read_text(encoding="utf-8") != generated:
            print("EL capability map 与声明漂移或被手改（§5.4 拒收）")
            return 1
        print("EL capability map 与生成结果一致")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(generated, encoding="utf-8")
    print(f"已生成 {OUT.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
