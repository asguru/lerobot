#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
from pathlib import Path

from termcolor import colored
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# Imports for FSDP
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
)

from lerobot.common.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.common.datasets.utils import load_json, write_json
from lerobot.common.optim.optimizers import load_optimizer_state, save_optimizer_state
from lerobot.common.optim.schedulers import load_scheduler_state, save_scheduler_state
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.random_utils import load_rng_state, save_rng_state
from lerobot.configs.train import TrainPipelineConfig


def log_output_dir(out_dir):
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {out_dir}")


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def save_training_step(step: int, save_dir: Path) -> None:
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


# def update_last_checkpoint(checkpoint_dir: Path) -> Path:
#     last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
#     if last_checkpoint_dir.is_symlink():
#         last_checkpoint_dir.unlink()
#     relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
#     last_checkpoint_dir.symlink_to(relative_target)


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    # Only create symlink on rank 0
    if dist.is_initialized():
        if dist.get_rank() == 0:
            last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
            if last_checkpoint_dir.is_symlink():
                last_checkpoint_dir.unlink()
            relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
            last_checkpoint_dir.symlink_to(relative_target)
        # Make sure all processes wait for symlink creation
        dist.barrier()
    else:
        # Original behavior for non-distributed case
        last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
        if last_checkpoint_dir.is_symlink():
            last_checkpoint_dir.unlink()
        relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
        last_checkpoint_dir.symlink_to(relative_target)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
) -> None:
    """This function creates the following directory structure:

    005000/  #  training step at checkpoint
    ├── pretrained_model/
    │   ├── config.json  # policy config
    │   ├── model.safetensors  # policy weights
    │   └── train_config.json  # train config
    └── training_state/
        ├── optimizer_param_groups.json  #  optimizer param groups
        ├── optimizer_state.safetensors  # optimizer state
        ├── rng_state.safetensors  # rng states
        ├── scheduler_state.json  # scheduler state
        └── training_step.json  # training step

    Args:
        cfg (TrainPipelineConfig): The training config used for this run.
        step (int): The training step at that checkpoint.
        policy (PreTrainedPolicy): The policy to save.
        optimizer (Optimizer | None, optional): The optimizer to save the state from. Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler to save the state from. Defaults to None.
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)
    cfg.save_pretrained(pretrained_dir)
    save_training_state(checkpoint_dir, step, optimizer, scheduler)


def save_fsdp_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
) -> None:
    """Creates the same directory structure as save_checkpoint but handles FSDP-specific saving:

    005000/  #  training step at checkpoint
    ├── pretrained_model/
    │   ├── config.json  # policy config
    │   ├── model.safetensors  # policy weights
    │   └── train_config.json  # train config
    └── training_state/
        ├── optimizer_param_groups.json  #  optimizer param groups
        ├── optimizer_state.safetensors  # optimizer state
        ├── rng_state.safetensors  # rng states
        ├── scheduler_state.json  # scheduler state
        └── training_step.json  # training step

    Args:
        cfg (TrainPipelineConfig): The training config used for this run.
        step (int): The training step at that checkpoint.
        policy (PreTrainedPolicy): The FSDP-wrapped policy to save.
        optimizer (Optimizer): The optimizer to save the state from.
        scheduler (LRScheduler | None, optional): The scheduler to save the state from.
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR

    # Only save on rank 0 to avoid conflicts
    if dist.get_rank() == 0:
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        training_state_dir.mkdir(parents=True, exist_ok=True)

    # Ensure all processes reach this point before saving
    dist.barrier()

    with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT):
        policy_state = policy.state_dict()
        if dist.get_rank() == 0:
            # Get the unwrapped model to access save_pretrained
            unwrapped_policy = policy.module
            unwrapped_policy.save_pretrained(pretrained_dir, state_dict=policy_state)
            cfg.save_pretrained(pretrained_dir)

    # Save training state
    if dist.get_rank() == 0:
        # Save step
        write_json({"step": step}, training_state_dir / "training_step.json")

        # Save RNG states
        rng_state = {
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        save_safetensors(rng_state, training_state_dir / "rng_state.safetensors")

        # Save optimizer state
        optim_state = FSDP.full_optim_state_dict(policy, optimizer)
        save_safetensors(optim_state, training_state_dir / "optimizer_state.safetensors")
        write_json(optimizer.state_dict()["param_groups"], training_state_dir / "optimizer_param_groups.json")

        # Save scheduler state if it exists
        if scheduler is not None:
            write_json(scheduler.state_dict(), training_state_dir / "scheduler_state.json")

    # Ensure all processes wait for saving to complete
    dist.barrier()


# def save_fsdp_checkpoint(checkpoint_dir, step, policy, optimizer, lr_scheduler, step, checkpoint_dir):
#     save_policy_name = f"{checkpoint_dir}/policy.pt"
#     save_optim_name = f"{checkpoint_dir}/optimizer.pt"
    
#     # Save model in FSDP format
#     full_state_dict = policy.state_dict()
    
#     if dist.get_rank() == 0:
#         torch.save(full_state_dict, save_policy_name)
#         # Save optimizer state
#         optim_state = FSDP.full_optim_state_dict(policy, optimizer)
#         torch.save({
#             'step': step,
#             'optimizer_state_dict': optim_state,
#             'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None
#         }, save_optim_name)
    
#     dist.barrier()


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> None:
    """
    Saves the training step, optimizer state, scheduler state, and rng state.

    Args:
        save_dir (Path): The directory to save artifacts to.
        train_step (int): Current training step.
        optimizer (Optimizer | None, optional): The optimizer from which to save the state_dict.
            Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler from which to save the state_dict.
            Defaults to None.
    """
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir)
    save_rng_state(save_dir)
    if optimizer is not None:
        save_optimizer_state(optimizer, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)


def load_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None
) -> tuple[int, Optimizer, LRScheduler | None]:
    """
    Loads the training step, optimizer state, scheduler state, and rng state.
    This is used to resume a training run.

    Args:
        checkpoint_dir (Path): The checkpoint directory. Should contain a 'training_state' dir.
        optimizer (Optimizer): The optimizer to load the state_dict to.
        scheduler (LRScheduler | None): The scheduler to load the state_dict to (can be None).

    Raises:
        NotADirectoryError: If 'checkpoint_dir' doesn't contain a 'training_state' dir

    Returns:
        tuple[int, Optimizer, LRScheduler | None]: training step, optimizer and scheduler with their
            state_dict loaded.
    """
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)

    return step, optimizer, scheduler
