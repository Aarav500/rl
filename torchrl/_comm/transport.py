# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Common payload-transport selection envelope for TorchRL services.

TorchRL separates three concerns when a service moves data between workers:

1. *Placement and lifecycle*: where the service owner runs
   (``service_backend``, see :mod:`torchrl._comm.backends`).
2. *Interaction pattern*: the domain protocol (request/reply, ordered
   commands, shared state, ...). Each domain keeps its own contract.
3. *Payload movement*: how the values referenced by an operation travel
   (``transport``). This module provides the shared configuration envelope
   for that third layer: :data:`TransportKind`, :class:`TransportConfig`,
   normalization helpers, and a per-domain resolver registry.

There is deliberately no common transport implementation ABC: queues, shared
memory, object stores, RPC, and collectives have different semantics. Domains
register a resolver that maps a :class:`TransportConfig` (plus domain context)
to their own concrete transport object.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast, Literal, TypeAlias

TransportKind: TypeAlias = Literal[
    "auto",
    "direct",
    "queue",
    "shared_memory",
    "ray",
    "rpc",
    "distributed",
    "monarch",
]

_TRANSPORT_KINDS = frozenset(
    {
        "auto",
        "direct",
        "queue",
        "shared_memory",
        "ray",
        "rpc",
        "distributed",
        "monarch",
    }
)


def normalize_transport_kind(kind: TransportKind | str | None) -> TransportKind:
    """Return the canonical spelling of a payload-transport kind.

    Args:
        kind: transport kind to normalize. ``None`` is a compatibility
            spelling for ``"auto"``.

    Returns:
        The canonical transport kind.

    Raises:
        ValueError: if the kind is unknown.

    Examples:
        >>> from torchrl._comm import normalize_transport_kind
        >>> normalize_transport_kind(None)
        'auto'
        >>> normalize_transport_kind("shared_memory")
        'shared_memory'
    """
    if kind is None:
        return "auto"
    if kind not in _TRANSPORT_KINDS:
        raise ValueError(
            f"Unsupported transport kind {kind!r}. "
            f"Expected one of {sorted(_TRANSPORT_KINDS)}."
        )
    return cast(TransportKind, kind)


@dataclass(frozen=True)
class TransportConfig:
    """Payload-transport selection for a TorchRL service.

    This is a small configuration envelope, not a transport implementation:
    ``kind`` names how payloads move (``"queue"``, ``"shared_memory"``,
    ``"ray"``, ...) independently of where the service owner runs
    (``service_backend``), and ``options`` carries kind-specific settings
    consumed by the domain resolver (e.g. ``num_slots``, ``request_spec``,
    and ``response_spec`` for the inference shared-memory transport).

    ``kind="auto"`` (default) lets the domain resolver pick the fastest safe
    transport from placement, topology, and the available payload contract;
    the resolved choice is deterministic and inspectable through the owning
    service's ``transport_kind`` property.

    Args:
        kind (TransportKind, optional): how payloads move. Defaults to
            ``"auto"``.
        options (dict[str, Any], optional): kind-specific options forwarded
            to the resolved transport constructor. Defaults to ``{}``.

    Examples:
        >>> from torchrl._comm import TransportConfig
        >>> TransportConfig()
        TransportConfig(kind='auto', options={})
        >>> config = TransportConfig("queue", {"use_manager": True})
        >>> config.kind
        'queue'
    """

    kind: TransportKind = "auto"
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", normalize_transport_kind(self.kind))
        object.__setattr__(self, "options", dict(self.options))


def as_transport_config(
    transport: TransportConfig | TransportKind | str | None = None,
    transport_options: dict[str, Any] | None = None,
) -> TransportConfig:
    """Normalize a user-facing transport choice into a :class:`TransportConfig`.

    Args:
        transport: ``None`` (compatibility spelling for ``"auto"``), a
            transport kind string, or an existing :class:`TransportConfig`.
        transport_options: kind-specific options. Only valid together with
            ``None`` or a string ``transport``; a :class:`TransportConfig`
            already carries its own options.

    Returns:
        A :class:`TransportConfig`.

    Raises:
        ValueError: if ``transport_options`` is combined with a
            :class:`TransportConfig`.
        TypeError: if ``transport`` is neither ``None``, a string, nor a
            :class:`TransportConfig`.

    Examples:
        >>> from torchrl._comm import as_transport_config
        >>> as_transport_config(None)
        TransportConfig(kind='auto', options={})
        >>> as_transport_config("queue", {"use_manager": True}).options
        {'use_manager': True}
    """
    if isinstance(transport, TransportConfig):
        if transport_options:
            raise ValueError(
                "transport_options cannot be combined with a TransportConfig; "
                "put the options in TransportConfig.options instead."
            )
        return transport
    if transport is None or isinstance(transport, str):
        return TransportConfig(
            normalize_transport_kind(transport), dict(transport_options or {})
        )
    raise TypeError(
        "transport must be None, a transport kind string, or a "
        f"TransportConfig; got {type(transport).__name__}. Concrete transport "
        "objects are domain-specific and should be passed to the domain API "
        "directly."
    )


# -- per-domain resolver registry ---------------------------------------------

_TRANSPORT_RESOLVERS: dict[str, Callable[..., Any]] = {}


def register_transport_resolver(
    domain: str, resolver: Callable[..., Any], *, overwrite: bool = False
) -> None:
    """Register the transport resolver for a service domain.

    A resolver maps a user-facing transport choice (``None``, kind string,
    :class:`TransportConfig`, or a domain-specific transport instance) plus
    domain context keyword arguments to a concrete transport object.

    Args:
        domain: the service domain name (e.g. ``"inference"``).
        resolver: the resolver callable.

    Keyword Args:
        overwrite (bool, optional): allow replacing an existing registration.
            Defaults to ``False``.

    Raises:
        ValueError: if the domain is already registered and ``overwrite`` is
            ``False``.
    """
    if domain in _TRANSPORT_RESOLVERS and not overwrite:
        raise ValueError(
            f"A transport resolver is already registered for domain "
            f"{domain!r}. Pass overwrite=True to replace it."
        )
    _TRANSPORT_RESOLVERS[domain] = resolver


def get_transport_resolver(domain: str) -> Callable[..., Any]:
    """Return the transport resolver registered for *domain*.

    Args:
        domain: the service domain name.

    Raises:
        KeyError: if no resolver is registered for the domain.
    """
    try:
        return _TRANSPORT_RESOLVERS[domain]
    except KeyError:
        raise KeyError(
            f"No transport resolver registered for domain {domain!r}. "
            f"Registered domains: {sorted(_TRANSPORT_RESOLVERS)}."
        ) from None


def resolve_transport(
    domain: str,
    transport: Any = None,
    **kwargs: Any,
) -> Any:
    """Resolve a transport choice through the registered domain resolver.

    Args:
        domain: the service domain name (e.g. ``"inference"``).
        transport: the user-facing transport choice, forwarded to the
            resolver.
        **kwargs: domain context forwarded to the resolver.

    Returns:
        The concrete transport object produced by the domain resolver.

    Examples:
        >>> from torchrl._comm import resolve_transport
        >>> import torchrl.modules.inference_server  # registers "inference"
        >>> transport = resolve_transport("inference", "queue")
        >>> type(transport).__name__
        'ThreadingTransport'
    """
    return get_transport_resolver(domain)(transport, **kwargs)
