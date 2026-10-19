"""BrandEval智能体模块（最小导出）。

为避免引入未安装的依赖（如 LLM 接口），仅导出基础可运行智能体：
- Brand
- Consumer
"""

from .brand import Brand
from .consumer import Consumer

__all__ = ["Brand", "Consumer"]