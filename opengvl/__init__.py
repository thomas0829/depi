"""Minimal opengvl package for instruction reward annotation."""

from opengvl.clients.qwen import QwenClient  # noqa: F401
from opengvl.metrics.instruction_reward import InstructionRewardResult  # noqa: F401

__all__ = ["QwenClient", "InstructionRewardResult"]
