"""RunConfig：Agent 流水线运行期公共开关的统一收口配置。

背景：use_llm / use_finbert / use_rule / rate_limit / use_semantic_cases /
max_documents 等开关此前在 SweepingOrchestrator 与各 Agent 的 __init__ 里
逐层透传，签名随开关增多而膨胀。

收口方式：
- 编排器（SweepingOrchestrator）统一构造 RunConfig 并下发给各 Agent；
- Agent 构造时「显式传入的参数优先」，未指定的公共开关从 RunConfig 读取；
- 各 Agent 特有的调参参数（top_k / horizons / model_dir 等）不进 RunConfig。
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ANNOUNCE_MAX_DOCUMENTS


@dataclass
class RunConfig:
    """Agent 流水线运行期开关。默认值 = 各 Agent 无参构造时的历史行为。"""

    use_llm: bool = True             # 公告研读/归因的 LLM 通道
    use_finbert: bool = True         # FinBERT 门控
    use_rule: bool = True            # 规则抽取通道（官方词典）
    rate_limit: float = 0.5          # 财务在线抓取限速（秒/请求）
    use_semantic_cases: bool = True  # 案例语义检索通道
    max_documents: int | None = ANNOUNCE_MAX_DOCUMENTS  # 公告研读最大文档数
    use_checkpointer: bool = False   # LangGraph 检查点（恢复用）
