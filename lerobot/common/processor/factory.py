#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

from .converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from .core import RobotAction, RobotObservation
from .pipeline import DataProcessorPipeline, IdentityProcessorStep, RobotProcessorPipeline

if TYPE_CHECKING:
    from lerobot.configs.policies import PreTrainedConfig


def make_default_teleop_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_action_processor


def make_default_robot_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return robot_action_processor


def make_default_robot_observation_processor() -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


def make_default_processors():
    teleop_action_processor = make_default_teleop_action_processor()
    robot_action_processor = make_default_robot_action_processor()
    robot_observation_processor = make_default_robot_observation_processor()
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    dataset_stats: dict[str, dict[str, Any]],
    device: torch.device | str | None = None,
) -> tuple[DataProcessorPipeline, DataProcessorPipeline]:
    """
    Creates preprocessor and postprocessor pipelines for a policy.
    
    This function creates the normalization processors needed for training and inference:
    - Preprocessor: Normalizes observations and actions before feeding to the policy
    - Postprocessor: Unnormalizes the policy's action outputs
    
    Args:
        policy_cfg: The policy configuration containing input/output features and
            normalization mapping.
        dataset_stats: The normalization statistics from the dataset (mean, std, 
            min, max, q01, q99, etc.).
        device: The target device for the processors.
    
    Returns:
        A tuple of (preprocessor, postprocessor) DataProcessorPipeline instances.
    
    Example:
        ```python
        from lerobot.common.processor.factory import make_pre_post_processors
        
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            dataset_stats=dataset.meta.stats,
            device="cuda"
        )
        
        # Save processors alongside the model
        preprocessor.save_pretrained(save_dir, config_filename="policy_preprocessor.json")
        postprocessor.save_pretrained(save_dir, config_filename="policy_postprocessor.json")
        ```
    """
    # Import here to avoid circular imports
    from .normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    
    # Combine input and output features
    features: dict[str, PolicyFeature] = {}
    if policy_cfg.input_features:
        features.update(policy_cfg.input_features)
    if policy_cfg.output_features:
        features.update(policy_cfg.output_features)
    
    # Get normalization mapping from policy config
    norm_map: dict[FeatureType, NormalizationMode] = {}
    if hasattr(policy_cfg, "normalization_mapping") and policy_cfg.normalization_mapping:
        for feature_type_str, mode_str in policy_cfg.normalization_mapping.items():
            try:
                feature_type = FeatureType(feature_type_str)
                norm_mode = NormalizationMode(mode_str)
                norm_map[feature_type] = norm_mode
            except (ValueError, KeyError):
                pass
    
    # Apply default normalization modes if not specified
    if FeatureType.VISUAL not in norm_map:
        norm_map[FeatureType.VISUAL] = NormalizationMode.IDENTITY
    if FeatureType.STATE not in norm_map:
        norm_map[FeatureType.STATE] = NormalizationMode.MEAN_STD
    if FeatureType.ACTION not in norm_map:
        norm_map[FeatureType.ACTION] = NormalizationMode.MEAN_STD
    
    # Create normalizer step for preprocessor
    normalizer_step = NormalizerProcessorStep(
        features=features,
        norm_map=norm_map,
        stats=dataset_stats,
        device=device,
    )
    
    # Create unnormalizer step for postprocessor (only for actions)
    action_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}
    unnormalizer_step = UnnormalizerProcessorStep(
        features=action_features,
        norm_map=norm_map,
        stats=dataset_stats,
        device=device,
    )
    
    # Create preprocessor pipeline
    preprocessor = DataProcessorPipeline(
        name="policy_preprocessor",
        steps=[normalizer_step],
    )
    
    # Create postprocessor pipeline
    postprocessor = DataProcessorPipeline(
        name="policy_postprocessor",
        steps=[unnormalizer_step],
    )
    
    return preprocessor, postprocessor

