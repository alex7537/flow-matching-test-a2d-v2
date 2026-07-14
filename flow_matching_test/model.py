"""Backward-compatible import for checkpoints and scripts using the old model path."""

from flow_matching_test.policies.base import SamplingResult
from flow_matching_test.policies.flow_matching import FlowMatchingPolicy

RGBConditionedFlowModel = FlowMatchingPolicy

__all__ = ["FlowMatchingPolicy", "RGBConditionedFlowModel", "SamplingResult"]
