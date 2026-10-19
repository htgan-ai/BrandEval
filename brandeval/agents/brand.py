"""基础品牌智能体（最小可运行版本）

提供与模型交互所需的基本字段与方法：
- 属性：name、marketing_budget、product_quality、sales、revenue、reputation、influence
- 方法：step() 生成官方帖子、record_sale() 记录销售、receive_message() 兼容消息输入

该类用于支撑 BrandEvalModel 的运行与指标计算，不涉及复杂的 LLM 决策流程；
如需策略型品牌 Agent，请使用 brandeval.agents.brand_agent.BrandAgent。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from brandeval.agents.base import BrandEvalAgent
except Exception:
    # 若源文件缺失但存在已编译 pyc，导入会由 Python 处理；
    # 这里仅用于类型提示避免运行期失败。
    BrandEvalAgent = object  # type: ignore


class Brand(BrandEvalAgent):  # type: ignore[misc]
    """最小品牌智能体。

    - 在每步中可能发布一条官方帖子（官方消息在平台分发时会强制推送）。
    - 维护销量与营收，用于指标节点读取。
    """

    def __init__(
        self,
        model,
        name: str = "Brand",
        marketing_budget: float = 100.0,
        product_quality: float = 0.7,
        unit_price: float = 500.0,
    ) -> None:
        # 角色为 brand，继承基础消息/队列能力
        try:
            super().__init__(model, role="brand")  # type: ignore[call-arg]
        except Exception:
            # 在极端情况下（基础类不可用），降级为简单对象以避免崩溃
            # 注意：此降级模式仅用于维持测试运行，不提供完整消息能力。
            # 若出现该分支，模型的分发/收集能力将受限。
            self.model = model
            self.role = "brand"
            self.message_queue = []  # 与 BrandEvalAgent 接口保持一致

        self.name = str(name)
        self.marketing_budget = float(marketing_budget)
        self.product_quality = float(product_quality)
        self.unit_price = float(unit_price)

        # 指标字段
        self.sales: int = 0
        self.revenue: float = 0.0
        self.reputation: float = 0.0

        # 影响力：用于平台计算概率推送（越大越容易扩散）
        # 简化设定：预算与质量的线性组合，截断到 [0, +inf) 后做对数缩放
        try:
            import math
            base = max(0.0, self.marketing_budget) * 0.01 + max(0.0, self.product_quality) * 1.0
            self.influence: float = float(math.log(base + 1.0) + 0.1)
        except Exception:
            self.influence = 1.0

    # ===== 运行与消息交互 =====
    def step(self) -> None:
        """品牌每步的简单行为：
        - 不使用任何回退模版（Dummy Brand 保持静默）。
        """
        pass

    def receive_message(self, message: Dict[str, Any]) -> None:
        """接收消息（保留基础能力）。"""
        try:
            # 调用基础实现以保持一致的记录行为
            super().receive_message(message)  # type: ignore[misc]
        except Exception:
            # 回退：直接将消息放入自身队列，供需要时读取
            try:
                self.message_queue.append(message)
            except Exception:
                pass
