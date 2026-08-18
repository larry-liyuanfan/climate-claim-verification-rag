from __future__ import annotations

import inspect
from typing import Any


def ensure_torch_pytree_compat(pytree_module: Any | None = None) -> bool:
    """Expose the public pytree registration API on PyTorch 2.1.

    Transformers 4.51+ calls ``register_pytree_node`` while the PyTorch 2.1
    module available on Spartan exposes the same operation under the legacy
    ``_register_pytree_node`` name.  The wrapper forwards only keyword
    arguments supported by the installed legacy function.  Newer PyTorch
    versions are left untouched.

    Returns ``True`` only when the compatibility alias was installed.
    """

    if pytree_module is None:
        import torch.utils._pytree as pytree_module

    if hasattr(pytree_module, "register_pytree_node"):
        return False
    legacy_register = getattr(pytree_module, "_register_pytree_node", None)
    if not callable(legacy_register):
        return False

    supported_keywords = set(inspect.signature(legacy_register).parameters)

    def register_pytree_node(
        node_type: Any,
        flatten_fn: Any,
        unflatten_fn: Any,
        **kwargs: Any,
    ) -> Any:
        forwarded = {key: value for key, value in kwargs.items() if key in supported_keywords}
        return legacy_register(node_type, flatten_fn, unflatten_fn, **forwarded)

    setattr(pytree_module, "register_pytree_node", register_pytree_node)
    return True
