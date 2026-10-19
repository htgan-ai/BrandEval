"""Independent rating of a public message's impact vector x_t.

Appendix C.1.3 of the paper maps each message to the impact signal X_t with a
regression fitted on human-rated data from lightweight text features.  That
fitted mapping is not part of this release.  The sandbox therefore rates every
official message with a rater that is independent of the agent that wrote it:

* with a configured model, a fixed-prompt rating call that returns three
  numbers in [-1, 1] (capability, value, emotion);
* otherwise a deterministic cue-lexicon rater (Chinese and English).

The acting agent's own ``predicted_impact`` is logged but never used for the
consumer update, so a policy cannot change the feedback it receives by
self-reporting.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, Optional

DIMENSIONS = ("capability", "value", "emotion")

# Cue lexicon.  Counts are capped per cue so that repetition cannot inflate a
# score; the mapping is tanh-bounded and scaled to a modest magnitude.
CUES: Dict[str, Dict[str, list]] = {
    "capability": {
        "positive": ["数据", "测试", "检测", "报告", "第三方", "权威", "认证", "标准", "工程师", "技术", "原因", "调查", "排查",
                     "修复", "升级", "OTA", "召回", "整改", "安全", "质量", "透明", "公开", "证据", "记录", "日志", "复现",
                     "evidence", "data", "test", "report", "audit", "fix", "recall", "engineering", "safety", "investigation"],
        "negative": ["谣言", "不实", "否认", "无关", "拒绝", "无法", "不便透露", "暂不", "无可奉告", "deny", "rumor", "refuse", "no comment"],
    },
    "value": {
        "positive": ["补偿", "赔偿", "赔付", "退款", "退换", "延保", "免费", "优惠", "权益", "保障", "承诺", "兑现", "服务", "售后",
                     "回购", "补贴", "refund", "compensation", "warranty", "free", "guarantee", "reimburse"],
        "negative": ["涨价", "加价", "收费", "自费", "概不", "不予", "免责", "不承担", "surcharge", "not liable", "no refund"],
    },
    "emotion": {
        "positive": ["抱歉", "道歉", "致歉", "歉意", "对不起", "理解", "感谢", "倾听", "真诚", "诚恳", "尊重", "关心", "陪伴", "家人",
                     "朋友们", "信任", "共同", "一起", "sorry", "apologize", "apology", "thank", "sincere", "listen", "together"],
        "negative": ["追究", "法律责任", "起诉", "造谣", "抹黑", "恶意", "指责", "攻击", "警告", "黑公关", "水军", "诽谤",
                     "lawsuit", "legal action", "malicious", "attack", "defamation", "smear"],
    },
}

RATING_PROMPT = (
    "You rate how a public message would change a typical consumer's view of the brand that posted it. "
    "Return only a JSON object with the keys capability, value and emotion, each a number in [-1, 1]: "
    "capability = perceived competence and safety, value = perceived value and fairness, "
    "emotion = emotional bond and trust. 0 means no change.\n\nMessage:\n"
)


def _count(text: str, cue: str, cap: int = 3) -> int:
    if not cue:
        return 0
    if re.fullmatch(r"[A-Za-z ]+", cue):
        return min(cap, len(re.findall(r"(?<![A-Za-z])" + re.escape(cue) + r"(?![A-Za-z])", text, flags=re.I)))
    return min(cap, text.count(cue))


class MessageImpactRater:
    """Rate a message's (capability, value, emotion) impact independently of its author."""

    def __init__(self, llm: Optional[Any] = None, scale: float = 0.5, slope: float = 0.35):
        self.llm = llm
        self.scale = float(scale)
        self.slope = float(slope)

    @classmethod
    def from_model_name(cls, model_name: Optional[str]) -> "MessageImpactRater":
        """Lexicon rater offline or without a model; otherwise a fixed-prompt LLM rater."""
        name = str(model_name or "").strip()
        # Same flag semantics as brandeval/gemini_data.py: network is disabled
        # unless the variable is exactly "0" (GEMINI_API_DISABLE_NETWORK is the fallback).
        disabled = os.environ.get("BRANDEVAL_API_DISABLE_NETWORK", os.environ.get("GEMINI_API_DISABLE_NETWORK", "1")) != "0"
        if not name or name.lower() == "offline" or disabled:
            return cls()
        try:
            from brandeval.llm.interface import GeminiChatLLM
            return cls(llm=GeminiChatLLM(model_name=name))
        except Exception:
            return cls()

    # ------------------------------------------------------------------
    def rate(self, text: str, author_role: str = "brand") -> Dict[str, float]:
        text = str(text or "").strip()
        if not text:
            return {key: 0.0 for key in DIMENSIONS}
        if self.llm is not None:
            rated = self._rate_with_llm(text)
            if rated is not None:
                return rated
        return self.rate_lexicon(text)

    def rate_lexicon(self, text: str) -> Dict[str, float]:
        impact: Dict[str, float] = {}
        for key in DIMENSIONS:
            pos = sum(_count(text, cue) for cue in CUES[key]["positive"])
            neg = sum(_count(text, cue) for cue in CUES[key]["negative"])
            impact[key] = round(self.scale * math.tanh(self.slope * (pos - neg)), 6)
        return impact

    def _rate_with_llm(self, text: str) -> Optional[Dict[str, float]]:
        try:
            raw = self.llm.generate(
                [{"role": "system", "content": RATING_PROMPT.split("\n\n")[0]}, {"role": "user", "content": text}],
                response_format="json", temperature=0.0, max_tokens=200,
            )
        except Exception:
            return None
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                return None
        if not isinstance(raw, dict):
            return None
        impact: Dict[str, float] = {}
        for key in DIMENSIONS:
            try:
                value = float(raw.get(key))
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value):
                return None
            impact[key] = max(-1.0, min(1.0, value))
        return impact
