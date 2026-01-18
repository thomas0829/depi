"""Minimal metrics package exposing instruction reward results."""

from opengvl.metrics.base import Metric, MetricResult  # noqa: F401
from opengvl.metrics.instruction_reward import InstructionRewardResult  # noqa: F401

__all__ = ["Metric", "MetricResult", "InstructionRewardResult"]
