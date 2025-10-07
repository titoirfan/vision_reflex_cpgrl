from __future__ import annotations
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from .unitree_a1_env import UnitreeA1Env


def resample_velocity_commands(
    env: UnitreeA1Env,
    env_ids: torch.Tensor,
):
    env.sample_new_commands(env_ids)
    # That's all folks
