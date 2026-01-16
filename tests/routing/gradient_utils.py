"""Gradient utility functions for tensor inspection.

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


def _should_skip_attr(key: str, skip: list[str], allowed_private: list[str] | None = None) -> bool:
    """Determine if an attribute should be skipped during traversal.

    Parameters
    ----------
    key : str
        The attribute name.
    skip : list[str]
        List of attribute names to skip.
    allowed_private : list[str], optional
        List of private attributes that should NOT be skipped.
        Default is ["_discharge_t", "_flat_indices", "_group_ids", "_scatter_input"].

    Returns
    -------
    bool
        True if the attribute should be skipped.
    """
    if allowed_private is None:
        allowed_private = ["_discharge_t", "_flat_indices", "_group_ids", "_scatter_input"]

    if key in skip:
        return True

    # Skip dunder methods/attributes (Python internals)
    if key.startswith("__") and key.endswith("__"):
        return True

    # Skip most private attributes, but allow specific ones we care about
    if key.startswith("_") and key not in allowed_private:
        return True

    return False


def find_gradient_tensors(
    obj: Any,
    depth: int = 0,
    max_depth: int = 25,
    required: bool = False,
    skip: list[str] | None = None,
    _seen: set[int] | None = None,
) -> Iterator[torch.Tensor]:
    """Find tensors in an object that could have gradients.

    Generator to find tensors associated with object which could have gradients,
    i.e. tensor objects that contain floating point values.

    Parameters
    ----------
    obj : Any
        Any python object which may contain tensors.
    depth : int, optional
        Current recursion depth, by default 0.
    max_depth : int, optional
        Maximum recursion depth, by default 25.
    required : bool, optional
        If True, only yield tensors that require gradients, by default False.
    skip : list[str], optional
        List of object attributes to skip, by default [].
    _seen : set[int], optional
        Set of object IDs already visited (for cycle detection).
        Should not be passed by caller; used internally.

    Yields
    ------
    torch.Tensor
        Tensors that are floating point and may have gradients.

    Examples
    --------
    >>> model = MyModel()
    >>> for tensor in find_gradient_tensors(model, required=True):
    ...     print(tensor.shape)
    """
    if skip is None:
        skip = []
    if _seen is None:
        _seen = set()

    # Stop recursion if max depth is reached
    if depth > max_depth:
        return

    # Prevent infinite recursion on circular references
    try:
        obj_id = id(obj)
    except TypeError:
        return

    if obj_id in _seen:
        return
    _seen.add(obj_id)

    # Skip types, modules, functions, and other non-data objects
    if isinstance(obj, type | type(lambda: None) | type(len)):
        return

    if isinstance(obj, torch.Tensor) and torch.is_floating_point(obj):
        if required:
            if obj.requires_grad:
                yield obj
        else:
            yield obj
    elif isinstance(obj, dict):
        for _key, value in obj.items():
            yield from find_gradient_tensors(value, depth + 1, max_depth, required, skip, _seen)
    elif isinstance(obj, list | tuple | torch.nn.ModuleList):
        for item in obj:
            yield from find_gradient_tensors(item, depth + 1, max_depth, required, skip, _seen)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for key, value in obj.__dict__.items():
            if _should_skip_attr(key, skip):
                continue
            yield from find_gradient_tensors(value, depth + 1, max_depth, required, skip, _seen)


def find_and_retain_grad(
    obj: Any,
    max_depth: int = 25,
    required: bool = False,
    skip: list[str] | None = None,
) -> None:
    """Find tensors in an object and ensure they retain gradients.

    Traverses the object hierarchy to find all floating point tensors
    and calls `retain_grad()` on them so gradients are preserved after
    backward pass.

    Parameters
    ----------
    obj : Any
        Any object which may contain tensors.
    max_depth : int, optional
        Maximum depth to recursively search objects, by default 25.
    required : bool, optional
        If True, only call `retain_grad()` on tensors which already have
        `requires_grad=True`, by default False.
    skip : list[str], optional
        List of object attributes to skip, by default [].

    Examples
    --------
    >>> model = MyModel()
    >>> find_and_retain_grad(model, required=True)
    >>> output = model(input)
    >>> output.backward()
    >>> # Now intermediate tensors have .grad populated
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
    parent_name: str | None = None,
    required: bool = False,
    skip: list[str] | None = None,
    _seen: set[int] | None = None,
) -> Iterator[str]:
    """Find names of tensors in an object.

    Generator to find the hierarchical names of tensors in an object,
    useful for debugging and identifying which tensors have or lack gradients.

    Parameters
    ----------
    obj : Any
        The object to inspect for tensors.
    name : str, optional
        The name of the current object, by default "Unknown".
    depth : int, optional
        Current recursion depth, by default 0.
    max_depth : int, optional
        Maximum recursion depth, by default 25.
    parent_name : str, optional
        The name of the parent object, by default None.
    required : bool, optional
        If True, only yield names of tensors that require gradients,
        by default False.
    skip : list[str], optional
        List of object attributes to skip, by default [].
    _seen : set[int], optional
        Set of object IDs already visited (for cycle detection).
        Should not be passed by caller; used internally.

    Yields
    ------
    str
        Names of tensors that are floating point and may have gradients.
        Names are formatted as hierarchical paths like "model.layer.weight".

    Examples
    --------
    >>> model = MyModel()
    >>> for name in get_tensor_names(model, name="model", required=True):
    ...     print(name)
    model.layer1.weight
    model.layer1.bias
    """
    if skip is None:
        skip = []
    if _seen is None:
        _seen = set()

    # Stop recursion if max depth is reached
    if depth > max_depth:
        return

    # Prevent infinite recursion on circular references
    try:
        obj_id = id(obj)
    except TypeError:
        return

    if obj_id in _seen:
        return
    _seen.add(obj_id)

    # Skip types, modules, functions, and other non-data objects
    if isinstance(obj, type | type(lambda: None) | type(len)):
        return

    if isinstance(obj, torch.Tensor) and torch.is_floating_point(obj):
        tensor_id = f"{parent_name}.{name}" if name[-1] != "]" else f"{parent_name}{name}"
        if required and not obj.requires_grad:
            return
        else:
            yield tensor_id
    elif isinstance(obj, dict):
        for key, value in obj.items():
            dict_id = f"{parent_name}.{name}"
            k = f"['{key}']"
            yield from get_tensor_names(
                value, k, depth + 1, max_depth, parent_name=dict_id, required=required, skip=skip, _seen=_seen
            )
    elif isinstance(obj, list | tuple | torch.nn.ModuleList):
        for i, item in enumerate(obj):
            list_id = f"{parent_name}.{name}" if name[-1] != "]" else f"{parent_name}{name}"
            idx = f"[{i}]"
            yield from get_tensor_names(
                item,
                idx,
                depth + 1,
                max_depth,
                parent_name=list_id,
                required=required,
                skip=skip,
                _seen=_seen,
            )
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for key, value in obj.__dict__.items():
            if _should_skip_attr(key, skip):
                continue
            k = f"{key}"
            n = f"{parent_name}.{name}" if parent_name else f"{name}"
            yield from get_tensor_names(
                value, k, depth + 1, max_depth, parent_name=n, required=required, skip=skip, _seen=_seen
            )


def print_grad_info(
    obj: Any,
    name: str = "Unknown",
    depth: int = 0,
    max_depth: int = 25,
    parent_name: str | None = None,
    required: bool = False,
    skip: list[str] | None = None,
    _seen: set[int] | None = None,
) -> None:
    """Print gradient information for tensors in the object.

    Traverses the object hierarchy and prints information about each
    tensor found, including whether it's a leaf tensor, requires gradients,
    and whether gradients exist.

    Parameters
    ----------
    obj : Any
        The object to inspect for tensors.
    name : str, optional
        The name of the current object, by default "Unknown".
    depth : int, optional
        Current recursion depth, by default 0.
    max_depth : int, optional
        Maximum recursion depth, by default 25.
    parent_name : str, optional
        The name of the parent object, by default None.
    required : bool, optional
        If True, only print info for tensors that require gradients,
        by default False.
    skip : list[str], optional
        List of object attributes to skip, by default [].
    _seen : set[int], optional
        Set of object IDs already visited (for cycle detection).
        Should not be passed by caller; used internally.

    Examples
    --------
    >>> model = MyModel()
    >>> loss = model(input).sum()
    >>> loss.backward()
    >>> print_grad_info(model, name="model", required=True)
       model.layer1.weight, Leaf, r: True, g: Exists
       model.layer1.bias, Leaf, r: True, g: Exists
    """
    if skip is None:
        skip = []
    if _seen is None:
        _seen = set()

    # Stop recursion if max depth is reached
    if depth > max_depth:
        return

    # Prevent infinite recursion on circular references
    try:
        obj_id = id(obj)
    except TypeError:
        return

    if obj_id in _seen:
        return
    _seen.add(obj_id)

    # Skip types, modules, functions, and other non-data objects
    if isinstance(obj, type | type(lambda: None) | type(len)):
        return

    if isinstance(obj, torch.Tensor):
        if required and not obj.requires_grad:
            return
        t = "Leaf" if obj.is_leaf else "Non-leaf"
        grad = "Exists" if obj.grad is not None else "None"
        tensor_id = f"{parent_name}.{name}" if name[-1] != "]" else f"{parent_name}{name}"
        print("  ", f"{tensor_id}, {t}, r: {obj.requires_grad}, g: {grad}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            dict_id = f"{parent_name}.{name}"
            k = f"['{key}']"
            print_grad_info(
                value, k, depth + 1, max_depth, parent_name=dict_id, required=required, skip=skip, _seen=_seen
            )
    elif isinstance(obj, list | tuple | torch.nn.ModuleList):
        for i, item in enumerate(obj):
            list_id = f"{parent_name}.{name}" if name[-1] != "]" else f"{parent_name}{name}"
            idx = f"[{i}]"
            print_grad_info(
                item,
                idx,
                depth + 1,
                max_depth,
                parent_name=list_id,
                required=required,
                skip=skip,
                _seen=_seen,
            )
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for key, value in obj.__dict__.items():
            if _should_skip_attr(key, skip):
                continue
            k = f"{key}"
            n = f"{parent_name}.{name}" if parent_name else f"{name}"
            print_grad_info(
                value, k, depth + 1, max_depth, parent_name=n, required=required, skip=skip, _seen=_seen
            )
