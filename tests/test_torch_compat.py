from __future__ import annotations

from types import SimpleNamespace

from climate_rag.torch_compat import ensure_torch_pytree_compat


def test_installs_legacy_alias_and_filters_new_keywords() -> None:
    calls: list[tuple[object, object, object, object]] = []

    def legacy_register(
        node_type: object,
        flatten_fn: object,
        unflatten_fn: object,
        *,
        to_dumpable_context: object = None,
    ) -> str:
        calls.append((node_type, flatten_fn, unflatten_fn, to_dumpable_context))
        return "registered"

    module = SimpleNamespace(_register_pytree_node=legacy_register)
    assert ensure_torch_pytree_compat(module) is True

    flatten = object()
    unflatten = object()
    result = module.register_pytree_node(
        dict,
        flatten,
        unflatten,
        to_dumpable_context="supported",
        serialized_type_name="ignored-on-torch-2.1",
    )
    assert result == "registered"
    assert calls == [(dict, flatten, unflatten, "supported")]


def test_leaves_public_api_untouched() -> None:
    public = object()
    module = SimpleNamespace(register_pytree_node=public)
    assert ensure_torch_pytree_compat(module) is False
    assert module.register_pytree_node is public
