"""
CUDA random stub module for MUSA platform.

This module exposes CUDA-style random API entry points under
`torch.cuda.random` and forwards them to `torch.musa` equivalents.
On a MUSA platform, behavior is provided by `torch.musa`; on non-MUSA
platforms, this module is only reachable when torchada patching is active.

Usage:
    import torchada  # Apply patches first
    import torch.cuda.random as cuda_random

    cuda_random.manual_seed(1234)
    state = cuda_random.get_rng_state()
"""

from typing import Iterable, List, Union

import torch
from torch import Tensor

__all__ = [
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
]


def get_rng_state(device: Union[int, str, torch.device] = "musa") -> Tensor:
    r"""Return the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'musa'`` for the current device.

    .. warning::
        This function eagerly initializes the backend device.
    """
    return torch.musa.get_rng_state(device)


def get_rng_state_all() -> List[Tensor]:
    r"""Return a list of ByteTensor representing the random number states of all devices."""
    return torch.musa.get_rng_state_all()


def set_rng_state(new_state: Tensor, device: Union[int, str, torch.device] = "musa") -> None:
    r"""Set the random number generator state of the specified GPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'musa'`` for the current device.
    """
    return torch.musa.set_rng_state(new_state, device)


def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    r"""Set the random number generator state of all devices.

    Args:
        new_states (Iterable of torch.ByteTensor): The desired state for each device.
    """
    return torch.musa.set_rng_state_all(new_states)


def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers for the current device.

    It's safe to call this function if the backend is unavailable; in that
    case, behavior is backend-defined.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    return torch.musa.manual_seed(seed)


def manual_seed_all(seed: int) -> None:
    r"""Set the seed for generating random numbers on all GPUs.

    It's safe to call this function if the backend is unavailable; in that
    case, behavior is backend-defined.

    Args:
        seed (int): The desired seed.
    """
    return torch.musa.manual_seed_all(seed)


def seed() -> None:
    r"""Set the seed for generating random numbers to a random number for the current GPU.

    It's safe to call this function if the backend is unavailable; in that
    case, behavior is backend-defined.
    """
    return torch.musa.seed()


def seed_all() -> None:
    r"""Set the seed for generating random numbers to a random number on all GPUs.

    It's safe to call this function if the backend is unavailable; in that
    case, behavior is backend-defined.
    """
    return torch.musa.seed_all()


def initial_seed() -> int:
    r"""Return the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes the backend device.
    """
    return torch.musa.initial_seed()
