# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Public transport resolution for the inference request/reply domain.

:func:`make_inference_transport` maps a user-facing transport choice
(``None``, a :data:`~torchrl._comm.TransportKind` string, a
:class:`~torchrl._comm.TransportConfig`, or an existing
:class:`~torchrl.modules.inference_server.InferenceTransport`) plus actor
topology context to a concrete transport instance. It backs the ``transport``
arguments of :class:`~torchrl.modules.inference_server.InferenceServer`,
:class:`~torchrl.modules.inference_server.ProcessInferenceServer`, and
:class:`~torchrl.collectors.AsyncBatchedCollector`, and is registered as the
``"inference"`` domain resolver in :mod:`torchrl._comm`.
"""
from __future__ import annotations

from typing import Any

from tensordict.base import TensorDictBase

from torchrl._comm import (
    as_transport_config,
    register_transport_resolver,
    TransportConfig,
    TransportKind,
)
from torchrl._utils import logger as torchrl_logger
from torchrl.modules.inference_server._monarch import MonarchTransport
from torchrl.modules.inference_server._mp import MPTransport
from torchrl.modules.inference_server._ray import RayTransport
from torchrl.modules.inference_server._shared_memory import SharedMemoryTransport
from torchrl.modules.inference_server._slot import SlotTransport
from torchrl.modules.inference_server._threading import ThreadingTransport
from torchrl.modules.inference_server._transport import InferenceTransport

_POLICY_BACKENDS = ("threading", "multiprocessing", "ray", "monarch")


def inference_transport_kind(transport: InferenceTransport) -> str:
    """Return the payload-transport kind of a concrete inference transport.

    This is the introspection counterpart of :func:`make_inference_transport`:
    it reports how a transport instance moves payloads, regardless of whether
    the instance was resolved from a kind string or injected directly.

    Args:
        transport (InferenceTransport): the transport instance.

    Returns:
        One of ``"direct"``, ``"queue"``, ``"shared_memory"``, ``"ray"``,
        ``"monarch"``, or ``"custom"`` for transports this module does not
        know about.

    Examples:
        >>> from torchrl.modules.inference_server import (
        ...     inference_transport_kind,
        ...     ThreadingTransport,
        ... )
        >>> inference_transport_kind(ThreadingTransport())
        'queue'
    """
    if isinstance(transport, SharedMemoryTransport):
        return "shared_memory"
    if isinstance(transport, SlotTransport):
        return "direct"
    if isinstance(transport, RayTransport):
        return "ray"
    if isinstance(transport, MonarchTransport):
        return "monarch"
    if isinstance(transport, (ThreadingTransport, MPTransport)):
        return "queue"
    return "custom"


def _resolve_auto(
    policy_backend: str,
    num_slots: int | None,
    request_spec: TensorDictBase | None,
    response_spec: TensorDictBase | None,
) -> TransportKind:
    """Pick the fastest safe transport kind for the given topology."""
    if policy_backend == "threading":
        # Fixed-producer thread topology: per-producer slots move direct
        # references with no lock or Future allocation.
        return "direct" if num_slots is not None else "queue"
    if policy_backend == "multiprocessing":
        if request_spec is not None and response_spec is not None:
            # Static TensorDict contract: preallocated CPU shared-memory
            # slots keep large payloads out of the queues.
            return "shared_memory"
        if request_spec is not None or response_spec is not None:
            raise ValueError(
                "transport='auto' received only one of request_spec/"
                "response_spec. Provide both to resolve to the shared-memory "
                "fast path, or neither to resolve to the queue transport."
            )
        return "queue"
    if policy_backend == "ray":
        return "ray"
    if policy_backend == "monarch":
        return "monarch"
    raise ValueError(
        f"Unknown policy_backend {policy_backend!r}. "
        f"Expected one of {_POLICY_BACKENDS}."
    )


def make_inference_transport(
    transport: InferenceTransport | TransportConfig | TransportKind | str | None = None,
    *,
    policy_backend: str | None = None,
    num_slots: int | None = None,
    request_spec: TensorDictBase | None = None,
    response_spec: TensorDictBase | None = None,
    transport_options: dict[str, Any] | None = None,
) -> InferenceTransport:
    """Resolve a transport choice into a concrete :class:`InferenceTransport`.

    The ``transport`` choice selects how payloads move; ``policy_backend``
    describes where the actors live. The two are independent: forcing a kind
    that is incompatible with the topology raises instead of silently
    degrading, and ``"auto"`` deterministically picks the fastest safe kind:

    - ``threading`` with ``num_slots`` known: :class:`SlotTransport`;
    - ``threading`` otherwise: :class:`ThreadingTransport`;
    - ``multiprocessing`` with ``request_spec`` and ``response_spec`` known:
      :class:`SharedMemoryTransport`;
    - ``multiprocessing`` without a payload contract: :class:`MPTransport`;
    - ``ray``: :class:`RayTransport`;
    - ``monarch``: :class:`MonarchTransport`.

    Args:
        transport: ``None`` (compatibility spelling for ``"auto"``), a
            :data:`~torchrl._comm.TransportKind` string, a
            :class:`~torchrl._comm.TransportConfig`, or an existing
            :class:`InferenceTransport` instance (returned unchanged).

    Keyword Args:
        policy_backend (str, optional): where the actors submitting requests
            live: one of ``"threading"``, ``"multiprocessing"``, ``"ray"``,
            or ``"monarch"``. ``None`` (default) resolves ``"auto"`` as
            in-process threads and skips the topology-compatibility check
            for forced kinds.
        num_slots (int, optional): number of producer slots / maximum
            in-flight requests. Consumed by :class:`SlotTransport` and
            :class:`SharedMemoryTransport`.
        request_spec (TensorDictBase, optional): representative single
            request, enabling the shared-memory fast path (see
            :class:`SharedMemoryTransport`).
        response_spec (TensorDictBase, optional): representative single
            response for the shared-memory fast path.
        transport_options (dict, optional): kind-specific options forwarded
            to the resolved transport constructor. Only valid with ``None``
            or string ``transport`` choices; a :class:`TransportConfig`
            carries its own options and a transport instance accepts none.
            ``num_slots``, ``request_spec``, and ``response_spec`` entries
            take precedence over the same-named keyword arguments.

    Returns:
        An :class:`InferenceTransport` instance.

    Raises:
        ValueError: for unknown kinds/backends, incompatible
            kind/backend combinations, or a forced ``"shared_memory"`` kind
            with an incomplete payload contract.

    Examples:
        >>> from torchrl.modules.inference_server import make_inference_transport
        >>> transport = make_inference_transport(
        ...     "queue", policy_backend="multiprocessing"
        ... )
        >>> type(transport).__name__
        'MPTransport'
        >>> import torch
        >>> from tensordict import TensorDict
        >>> transport = make_inference_transport(
        ...     policy_backend="multiprocessing",
        ...     num_slots=8,
        ...     request_spec=TensorDict({"observation": torch.zeros(4)}),
        ...     response_spec=TensorDict({"action": torch.zeros(2)}),
        ... )
        >>> type(transport).__name__
        'SharedMemoryTransport'
    """
    if isinstance(transport, InferenceTransport):
        if transport_options:
            raise ValueError(
                "transport_options cannot be combined with a transport "
                "instance; configure the instance directly."
            )
        return transport
    config = as_transport_config(transport, transport_options)
    options = dict(config.options)
    num_slots = options.pop("num_slots", num_slots)
    request_spec = options.pop("request_spec", request_spec)
    response_spec = options.pop("response_spec", response_spec)
    backend = policy_backend
    if backend is not None and backend not in _POLICY_BACKENDS:
        raise ValueError(
            f"Unknown policy_backend {backend!r}. "
            f"Expected one of {_POLICY_BACKENDS}."
        )

    kind = config.kind
    if kind == "auto":
        kind = _resolve_auto(
            backend if backend is not None else "threading",
            num_slots,
            request_spec,
            response_spec,
        )

    if kind == "direct":
        if backend not in (None, "threading"):
            raise ValueError(
                "transport='direct' moves object references and requires "
                f"in-process actors (policy_backend='threading'), got "
                f"policy_backend={backend!r}."
            )
        if num_slots is None:
            raise ValueError(
                "transport='direct' requires num_slots (one slot per actor "
                "thread) to build a SlotTransport."
            )
        resolved: InferenceTransport = SlotTransport(num_slots, **options)
    elif kind == "queue":
        if backend in (None, "threading"):
            resolved = ThreadingTransport(**options)
        elif backend == "multiprocessing":
            resolved = MPTransport(**options)
        else:
            raise ValueError(
                f"transport='queue' is not supported with "
                f"policy_backend={backend!r}; use transport='auto' or "
                f"transport={backend!r}."
            )
    elif kind == "shared_memory":
        if backend == "threading":
            raise ValueError(
                "transport='shared_memory' is redundant for "
                "policy_backend='threading': thread actors already share the "
                "process address space. Use transport='auto' or "
                "transport='direct'."
            )
        if backend not in (None, "multiprocessing"):
            raise ValueError(
                "transport='shared_memory' requires same-node actors "
                f"(policy_backend 'multiprocessing'), got "
                f"policy_backend={backend!r}."
            )
        missing = [
            name
            for name, value in (
                ("request_spec", request_spec),
                ("response_spec", response_spec),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "transport='shared_memory' requires a static payload "
                f"contract; missing {missing}. Pass them as keyword "
                "arguments or through transport_options."
            )
        if num_slots is None:
            raise ValueError(
                "transport='shared_memory' requires num_slots (the maximum "
                "number of concurrently in-flight requests)."
            )
        resolved = SharedMemoryTransport(
            request_spec, response_spec, num_slots=num_slots, **options
        )
    elif kind == "ray":
        if backend not in (None, "ray"):
            raise ValueError(
                f"transport='ray' is not supported with " f"policy_backend={backend!r}."
            )
        resolved = RayTransport(**options)
    elif kind == "monarch":
        if backend not in (None, "monarch"):
            raise ValueError(
                f"transport='monarch' is not supported with "
                f"policy_backend={backend!r}."
            )
        resolved = MonarchTransport(**options)
    else:  # "rpc", "distributed"
        raise ValueError(
            f"transport={kind!r} is not implemented for the inference "
            "domain. Supported kinds: 'auto', 'direct', 'queue', "
            "'shared_memory', 'ray', 'monarch'."
        )
    torchrl_logger.debug(
        f"Resolved inference transport {config.kind!r} with "
        f"policy_backend={backend!r} to {type(resolved).__name__}."
    )
    return resolved


register_transport_resolver("inference", make_inference_transport)
