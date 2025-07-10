"""
@author Nels Frazier

@date July 9 2025
@version 0.1

A set of utility functions to find tensors in an object that may have gradients.
An object can be any of the following container objects:
list, tuple, torch.nn.ModuleList, dict, or any object containing a __dict__ attribute
This is useful for debugging and ensuring that tensors retain gradients.

"""

from collections.abc import Iterator
from typing import Any

import torch


def find_gradient_tensors(
    obj: Any, depth=0, max_depth=25, required=False, skip: list[str] | None = None
) -> Iterator[torch.Tensor]:
    """Generator to find tensors associated with object which could have gradients,
    i.e. tensor objects that contain floating point values.

    Args:
        obj (Any): Any python object which may contain tensors.
        depth (int, optional): Current recursion depth. Defaults to 0.
        max_depth (int, optional): Maximum recursion depth. Defaults to 25.
        required (bool, optional): If True, only yield tensors that require gradients. Defaults to False.
        skip (list[str], optional): List of object attributes to skip. Defaults to [].
    Yields:
        torch.Tensor: Tensors that are floating point and may have gradients.
    """
    if skip is None:
        skip = []
    if depth > max_depth:
        yield from ()  # Stop recursion if max depth is reached
    if isinstance(obj, torch.Tensor) and torch.is_floating_point(obj):
        if required:
            if obj.requires_grad:
                yield obj
        else:
            yield obj
    elif isinstance(obj, dict):
        for _key, value in obj.items():
            yield from find_gradient_tensors(value, depth + 1, max_depth, required, skip)
    elif isinstance(obj, list | tuple | torch.nn.ModuleList):
        for item in obj:
            yield from find_gradient_tensors(item, depth + 1, max_depth, required, skip)
    elif hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            if key in skip:
                continue
            yield from find_gradient_tensors(value, depth + 1, max_depth, required, skip)
    else:
        # Handle other types if necessary
        pass


def find_and_retain_grad(obj: Any, max_depth=25, required=False, skip: list[str] | None = None) -> None:
    """Find tensors in an object and ensure they retain gradients.

    Args:
        obj (Any): Any object which may contain tensors.
        max_depth (int, optional): Max depth the recursively search objects. Defaults to 25.
        required (bool, optional): Only retaion_grad() on tensors which are marked with requires_grad. Defaults to False.
    """
    if skip is None:
        skip = []
    for tensor in find_gradient_tensors(obj, max_depth=max_depth, required=required, skip=skip):
        tensor.requires_grad_(True)  # Ensure requires_grad is set
        tensor.retain_grad()  # Retain gradient for all tensors


def get_tensor_names(
    obj: Any,
    name: str = "Unknown",
    depth: int = 0,
    max_depth: int = 25,
    parent_name: str = None,
    required: bool = False,
    skip: list[str] | None = None,
) -> Iterator[str]:
    """Generator to find names of tensors in an object.
    Args:
        obj (Any): The object to inspect for tensors.
        name (str, optional): The name of the current object. Defaults to "Unknown".
        depth (int, optional): Current recursion depth. Defaults to 0.
        max_depth (int, optional): Maximum recursion depth. Defaults to 25.
        parent_name (str, optional): The name of the parent object. Defaults to None.
        required (bool, optional): If True, only yield names of tensors that require gradients.
        skip (list[str], optional): List of object attributes to skip. Defaults to [].
    Yields:
        str: Names of tensors that are floating point and may have gradients.
    """
    if skip is None:
        skip = []
    if depth > max_depth:
        return
    if isinstance(obj, torch.Tensor) and torch.is_floating_point(obj):
        id = f"{parent_name}.{name}" if name[-1] != "]" else f"{parent_name}{name}"
        if required and not obj.requires_grad:
            return
        else:
            yield id
    elif isinstance(obj, dict):
        for key, value in obj.items():
            id = f"{parent_name}.{name}"
            k = f"['{key}']"
            yield from get_tensor_names(
                value, k, depth + 1, max_depth, parent_name=id, required=required, skip=skip
            )
    elif isinstance(obj, list | tuple | torch.nn.ModuleList):
        for i, item in enumerate(obj):
            id = f"{parent_name}.{name}" if name[-1] != "]" else f"{parent_name}{name}"
            idx = f"[{i}]"
            yield from get_tensor_names(
                item, idx, depth + 1, max_depth, parent_name=id, required=required, skip=skip
            )
    elif hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            if key in skip:
                continue
            k = f"{key}"
            n = f"{parent_name}.{name}" if parent_name else f"{name}"
            yield from get_tensor_names(
                value, k, depth + 1, max_depth, parent_name=n, required=required, skip=skip
            )


def print_grad_info(
    obj: Any,
    name: str = "Unknown",
    depth: int = 0,
    max_depth: int = 25,
    parent_name: str = None,
    required: bool = False,
    skip: list[str] | None = None,
) -> None:
    """Print gradient information for tensors in the object.

    Args:
        obj (Any): The object to inspect for tensors.
        name (str, optional): The name of the current object. Defaults to "Unknown".
        depth (int, optional): Current recursion depth. Defaults to 0.
        max_depth (int, optional): Maximum recursion depth. Defaults to 25.
        parent_name (str, optional): The name of the parent object. Defaults to None.
        skip (list[str], optional): List of object attributes to skip. Defaults to [].
    """
    if skip is None:
        skip = []
    if depth > max_depth:
        return
    if isinstance(obj, torch.Tensor):
        if required and not obj.requires_grad:
            return
        t = "Leaf" if obj.is_leaf else "Non-leaf"
        grad = "Exists" if obj.grad is not None else "None"
        id = f"{parent_name}.{name}" if name[-1] != "]" else f"{parent_name}{name}"
        print("  ", f"{id}, {t}, r: {obj.requires_grad}, g: {grad}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            # print(f"Key: {key}")
            id = f"{parent_name}.{name}"
            # id = f"{parent_name}.{name}" if name[-1] != "]" else f"{parent_name}{name}"
            k = f"['{key}']"
            print_grad_info(value, k, depth + 1, max_depth, parent_name=id, required=required, skip=skip)
    elif isinstance(obj, list | tuple | torch.nn.ModuleList):
        for i, item in enumerate(obj):
            id = f"{parent_name}...{name}"
            id = f"{parent_name}.{name}" if name[-1] != "]" else f"{parent_name}{name}"
            idx = f"[{i}]"
            print_grad_info(item, idx, depth + 1, max_depth, parent_name=id, required=required, skip=skip)
    elif hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            if key in skip:
                continue
            k = f"{key}"
            n = f"{parent_name}.{name}" if parent_name else f"{name}"
            print_grad_info(value, k, depth + 1, max_depth, parent_name=n, required=required, skip=skip)
