# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Inference transport over torch.distributed point-to-point primitives.

:class:`DistributedTransport` serves actors that live in other processes or
on other nodes (e.g. Ray actors) without touching the *default* process
group: every server/client pair gets standalone
:class:`~torch.distributed.ProcessGroupGloo` (and, for CUDA payloads,
:class:`~torch.distributed.ProcessGroupNCCL`) objects rendezvoused through a
transport-owned :class:`~torch.distributed.TCPStore`, so training code
remains free to use ``init_process_group`` for its own collectives.

Control messages (request/response headers, exceptions) always travel over
gloo; fixed-spec tensor payloads travel over gloo (CPU) or NCCL (GPU-direct)
depending on ``backend``. NCCL receives are only posted after a control
header announces the payload, which keeps the NCCL watchdog away from idle
receives and keeps each communicator single-threaded.
"""
from __future__ import annotations

import pickle
import queue
import socket
import threading
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from tensordict.base import _is_leaf_nontensor, TensorDictBase
from tensordict.utils import NestedKey

from torchrl._utils import logger as torchrl_logger
from torchrl.modules.inference_server._transport import InferenceTransport

_TAG_HEADER = 0
_TAG_PAYLOAD_BASE = 1

# Request header: (req_id, 0). Response header: (req_id, status) with
# status 0 = ok, 1 = exception (pickled under the store key _exc_key).
_HEADER_NUMEL = 2

_STATUS_OK = 0
_STATUS_EXC = 1

_CLIENT_COUNT_KEY = "distributed_transport/num_clients"

# Point-to-point op timeout. Deliberately much larger than the rendezvous
# timeout: header receives block idle for as long as actors stay quiet, and
# an op-timeout abort racing an arriving message could lose it. The retry
# loops below are a fallback, not the idle mechanism.
_OP_TIMEOUT_S = 86_400.0


def _exc_key(client_idx: int, req_id: int) -> str:
    return f"distributed_transport/exc/{client_idx}/{req_id}"


def _sorted_leaf_keys(spec: TensorDictBase) -> list[NestedKey]:
    # _is_leaf_nontensor surfaces NonTensorData leaves (excluded by the
    # default leaf iterator) so validation rejects them instead of ignoring
    # them; the keys are sorted so both peers derive the same wire order.
    keys = list(
        spec.keys(include_nested=True, leaves_only=True, is_leaf=_is_leaf_nontensor)
    )
    return sorted(keys, key=lambda k: (k,) if isinstance(k, str) else tuple(k))


def _validate_spec(spec: TensorDictBase, argname: str, backend: str) -> list[NestedKey]:
    device_type = "cuda" if backend == "nccl" else "cpu"
    keys = _sorted_leaf_keys(spec)
    if not keys:
        raise ValueError(f"{argname} must contain at least one tensor leaf.")
    for key in keys:
        value = spec.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"DistributedTransport specs only support tensor leaves; "
                f"{argname} has a {type(value).__name__} at key {key!r}."
            )
        if value.device.type != device_type:
            raise ValueError(
                f"DistributedTransport(backend={backend!r}) requires "
                f"{device_type} spec tensors; {argname} has a "
                f"{value.device} tensor at key {key!r}."
            )
    return keys


def _retry_while(stop_event: threading.Event, fn, description: str):
    """Run *fn* until it succeeds, retrying on rendezvous/op timeouts."""
    while not stop_event.is_set():
        try:
            return fn()
        except RuntimeError as err:  # noqa: PERF203
            message = str(err).lower()
            if "timed out" in message or "timeout" in message:
                continue
            torchrl_logger.warning(
                f"DistributedTransport: {description} failed with {err!r}."
            )
            raise
    return None


class _PairEndpoint:
    """One side of a server/client pair: control (gloo) + data groups."""

    def __init__(
        self,
        store,
        client_idx: int,
        rank: int,
        backend: str,
        timeout: float,
        device: torch.device,
    ):
        base = f"distributed_transport/pair/{client_idx}"
        pg_timeout = timedelta(seconds=_OP_TIMEOUT_S)
        # Directional groups: one per (direction, plane) so that each group
        # is only ever used by a single thread on each side (required for
        # NCCL, hygienic for gloo).
        self.ctrl_req = dist.ProcessGroupGloo(
            dist.PrefixStore(f"{base}/ctrl_req", store), rank, 2, pg_timeout
        )
        self.ctrl_resp = dist.ProcessGroupGloo(
            dist.PrefixStore(f"{base}/ctrl_resp", store), rank, 2, pg_timeout
        )
        if backend == "nccl":
            self.data_req = dist.ProcessGroupNCCL(
                dist.PrefixStore(f"{base}/data_req", store), rank, 2
            )
            self.data_resp = dist.ProcessGroupNCCL(
                dist.PrefixStore(f"{base}/data_resp", store), rank, 2
            )
        else:
            self.data_req = self.ctrl_req
            self.data_resp = self.ctrl_resp
        self.device = device

    def new_header(self) -> torch.Tensor:
        return torch.zeros(_HEADER_NUMEL, dtype=torch.int64)

    def send(self, pg, tensors: list[torch.Tensor], peer: int) -> None:
        for tag_offset, tensor in enumerate(tensors):
            pg.send([tensor], peer, _TAG_PAYLOAD_BASE + tag_offset).wait()

    def recv(self, pg, tensors: list[torch.Tensor], peer: int) -> None:
        for tag_offset, tensor in enumerate(tensors):
            pg.recv([tensor], peer, _TAG_PAYLOAD_BASE + tag_offset).wait()


class _DistributedFuture:
    """Future for one in-flight :class:`DistributedTransport` request."""

    _MISSING = object()

    def __init__(self, client: _DistributedClient, req_id: int):
        self._client = client
        self._req_id = req_id
        self._outcome = self._MISSING

    def done(self) -> bool:
        """Return ``True`` when the result can be read without blocking."""
        if self._outcome is not self._MISSING:
            return True
        try:
            self._outcome = self._client._get_reply(self._req_id, timeout=0)
        except queue.Empty:
            return False
        return True

    def result(self, timeout: float | None = None) -> TensorDictBase:
        """Return the inference result or raise its remote exception.

        Raises :class:`queue.Empty` when *timeout* elapses before the server
        replies; the request stays in flight and ``result`` can be retried.
        """
        if self._outcome is self._MISSING:
            self._outcome = self._client._get_reply(self._req_id, timeout=timeout)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class _DistributedClient:
    """Actor-side client for :class:`DistributedTransport`.

    Picklable: carries only the store coordinates, its client index, and the
    payload specs. The pair process groups are built lazily in the actor
    process on first use; the first request therefore blocks until the
    server side has connected the pair.
    """

    def __init__(
        self,
        store_info: tuple[str, int],
        client_idx: int,
        request_spec: TensorDictBase,
        response_spec: TensorDictBase,
        backend: str,
        timeout: float,
        device: torch.device,
    ):
        self._store_info = store_info
        self._client_idx = client_idx
        self._request_spec = request_spec
        self._response_spec = response_spec
        self._backend = backend
        self._timeout = timeout
        self._device = device
        self._request_keys = _sorted_leaf_keys(request_spec)
        self._response_keys = _sorted_leaf_keys(response_spec)
        self._endpoint: _PairEndpoint | None = None
        self._lock = threading.Lock()
        self._next_req_id = 0
        self._reply_queue: queue.Queue | None = None
        self._buffered: dict[int, object] = {}
        self._reply_lock = threading.Lock()
        self._stop_event: threading.Event | None = None

    @property
    def client_id(self) -> int:
        """The client index assigned by the owning transport."""
        return self._client_idx

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_endpoint"] = None
        state["_lock"] = None
        state["_reply_queue"] = None
        state["_reply_lock"] = None
        state["_stop_event"] = None
        state["_buffered"] = {}
        for post_connect in ("_store", "_send_buffer", "_recv_buffer"):
            state.pop(post_connect, None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._reply_lock = threading.Lock()

    def _ensure_connected(self) -> None:
        if self._endpoint is not None:
            return
        host, port = self._store_info
        store = dist.TCPStore(
            host, port, is_master=False, timeout=timedelta(seconds=self._timeout)
        )
        self._store = store
        self._endpoint = _PairEndpoint(
            store,
            self._client_idx,
            rank=1,
            backend=self._backend,
            timeout=self._timeout,
            device=self._device,
        )
        self._send_buffer = self._request_spec.clone().to(self._device)
        self._recv_buffer = self._response_spec.clone().to(self._device)
        self._reply_queue = queue.Queue()
        self._stop_event = threading.Event()
        receiver = threading.Thread(
            target=self._receive_loop,
            daemon=True,
            name=f"DistributedTransport-client-{self._client_idx}",
        )
        receiver.start()

    def _receive_loop(self) -> None:
        endpoint = self._endpoint
        header = endpoint.new_header()
        recv_leaves = [self._recv_buffer.get(key) for key in self._response_keys]
        while not self._stop_event.is_set():
            work = _retry_while(
                self._stop_event,
                lambda: endpoint.ctrl_resp.recv([header], 0, _TAG_HEADER).wait(),
                "response header recv",
            )
            if work is None and self._stop_event.is_set():
                return
            req_id, status = int(header[0]), int(header[1])
            if status == _STATUS_OK:
                endpoint.recv(endpoint.data_resp, recv_leaves, 0)
                self._reply_queue.put((req_id, self._recv_buffer.clone()))
            else:
                raw = self._store.get(_exc_key(self._client_idx, req_id))
                try:
                    exc = pickle.loads(raw)
                except Exception as err:  # pragma: no cover
                    exc = RuntimeError(f"Remote exception (unpicklable): {err!r}")
                self._reply_queue.put((req_id, exc))

    def _get_reply(self, req_id: int, timeout: float | None = None):
        with self._reply_lock:
            if req_id in self._buffered:
                return self._buffered.pop(req_id)
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                remaining = (
                    None if deadline is None else max(deadline - time.monotonic(), 0)
                )
                reply_id, payload = self._reply_queue.get(timeout=remaining)
                if reply_id == req_id:
                    return payload
                self._buffered[reply_id] = payload

    def submit(self, td: TensorDictBase) -> _DistributedFuture:
        """Send a request to the server and return a future for the reply."""
        with self._lock:
            self._ensure_connected()
            request = td.select(*self._request_keys, strict=True)
            req_id = self._next_req_id
            self._next_req_id += 1
            self._send_buffer.update_(request)
            endpoint = self._endpoint
            header = torch.tensor([req_id, 0], dtype=torch.int64)
            endpoint.ctrl_req.send([header], 0, _TAG_HEADER).wait()
            send_leaves = [self._send_buffer.get(key) for key in self._request_keys]
            endpoint.send(endpoint.data_req, send_leaves, 0)
        return _DistributedFuture(self, req_id)

    def __call__(
        self, td: TensorDictBase, timeout: float | None = None
    ) -> TensorDictBase:
        """Submit a request and block for its result."""
        return self.submit(td).result(timeout=timeout)


class DistributedTransport(InferenceTransport):
    """Cross-process/cross-node transport over torch.distributed p2p.

    Serves actors that cannot share memory or multiprocessing queues with
    the server -- typically Ray actors, possibly on other nodes. Each client
    gets its own standalone server/client pair of process groups
    rendezvoused through a transport-owned
    :class:`~torch.distributed.TCPStore`; the *default* process group is
    never touched, so training code can keep using ``init_process_group``
    for its own collectives.

    Small request/response headers (and pickled exceptions) travel over
    gloo control groups. Fixed-spec tensor payloads travel over gloo
    (``backend="gloo"``, CPU tensors) or NCCL (``backend="nccl"``, CUDA
    tensors, GPU-direct -- no host round trip).

    As with :class:`SharedMemoryTransport`, the payload contract is static:
    the specs fix the transmitted keys, shapes, and dtypes, extra keys are
    dropped, and a missing declared key raises.

    Args:
        request_spec (TensorDictBase): a representative single request.
        response_spec (TensorDictBase): a representative single response
            (including server-added keys to forward, e.g.
            ``"policy_version"``).

    Keyword Args:
        backend (str, optional): payload backend, ``"gloo"`` (CPU specs) or
            ``"nccl"`` (CUDA specs). Control headers always use gloo.
            Defaults to ``"gloo"``.
        timeout (float, optional): rendezvous and store timeout in seconds.
            Defaults to ``300.0``.
        host (str, optional): interface for the rendezvous store. Defaults
            to the local hostname address.
        port (int, optional): port for the rendezvous store. Defaults to an
            OS-assigned free port.

    .. note::
        Clients are created with :meth:`client` and are picklable; the pair
        process groups are built lazily in the actor process, so the first
        request blocks until the server side is running. The server
        discovers clients through the store, so clients may be created
        before or after the server starts.

    .. note::
        Requests are copied into fixed staging buffers on both sides (one
        payload copy per direction); nothing is pickled on the hot path.

    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.inference_server import (
        ...     DistributedTransport,
        ...     InferenceServer,
        ... )
        >>> request_spec = TensorDict({"observation": torch.zeros(4)})
        >>> response_spec = TensorDict({"action": torch.zeros(2)})
        >>> transport = DistributedTransport(request_spec, response_spec)
        >>> client = transport.client()  # picklable; ship to a Ray actor
        >>> policy = TensorDictModule(
        ...     torch.nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
        ... )
        >>> with InferenceServer(policy, transport, max_batch_size=4):
        ...     result = client(TensorDict({"observation": torch.randn(4)}))
        >>> assert result["action"].shape == (2,)
    """

    def __init__(
        self,
        request_spec: TensorDictBase,
        response_spec: TensorDictBase,
        *,
        backend: str = "gloo",
        timeout: float = 300.0,
        host: str | None = None,
        port: int | None = None,
    ):
        if not dist.is_available():
            raise RuntimeError(
                "torch.distributed is not available; DistributedTransport "
                "requires a torch build with distributed support."
            )
        if backend not in ("gloo", "nccl"):
            raise ValueError(
                f"Unsupported DistributedTransport backend {backend!r}. "
                "Expected 'gloo' or 'nccl'."
            )
        device_type = "cuda" if backend == "nccl" else "cpu"
        self._request_keys = _validate_spec(request_spec, "request_spec", backend)
        self._response_keys = _validate_spec(response_spec, "response_spec", backend)
        self._request_spec = request_spec.clone()
        self._response_spec = response_spec.clone()
        self._backend = backend
        self._timeout = timeout
        self._device = torch.device(device_type)

        if host is None:
            host = socket.gethostbyname(socket.gethostname())
        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("", 0))
                port = sock.getsockname()[1]
        self._store_info = (host, port)
        # The owner keeps the master store alive; unpickled copies (e.g. a
        # ProcessInferenceServer child) reconnect as store clients.
        self._store = dist.TCPStore(
            host,
            port,
            is_master=True,
            timeout=timedelta(seconds=timeout),
            wait_for_workers=False,
        )
        self._store.add(_CLIENT_COUNT_KEY, 0)

        self._ready_queue: queue.Queue | None = None
        self._peeked = None
        self._stop_event: threading.Event | None = None
        self._listener: threading.Thread | None = None
        self._endpoints: dict[int, _PairEndpoint] = {}
        self._response_buffers: dict[int, TensorDictBase] = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Server-side runtime state is rebuilt in the process that runs the
        # serve loop; the master store handle stays with the owner.
        state["_store"] = None
        state["_ready_queue"] = None
        state["_peeked"] = None
        state["_stop_event"] = None
        state["_listener"] = None
        state["_endpoints"] = {}
        state["_response_buffers"] = {}
        return state

    def _get_store(self):
        if self._store is None:
            host, port = self._store_info
            self._store = dist.TCPStore(
                host,
                port,
                is_master=False,
                timeout=timedelta(seconds=self._timeout),
            )
        return self._store

    # -- actor API ------------------------------------------------------------

    def client(self) -> _DistributedClient:
        """Create a picklable actor-side client.

        Each call reserves a new client index through the rendezvous store,
        so clients can be created from the owner or from an unpickled copy
        of the transport.
        """
        client_idx = int(self._get_store().add(_CLIENT_COUNT_KEY, 1)) - 1
        return _DistributedClient(
            self._store_info,
            client_idx,
            self._request_spec,
            self._response_spec,
            self._backend,
            self._timeout,
            self._device,
        )

    def submit(self, td: TensorDictBase):
        """Not supported -- use :meth:`client` to obtain an actor handle."""
        raise RuntimeError(
            "DistributedTransport.submit() is not supported. "
            "Call transport.client() to create a client."
        )

    # -- server side ------------------------------------------------------------

    def _ensure_server(self) -> None:
        if self._listener is not None:
            return
        self._ready_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._listener = threading.Thread(
            target=self._listen_for_clients,
            daemon=True,
            name="DistributedTransport-listener",
        )
        self._listener.start()

    def _listen_for_clients(self) -> None:
        store = self._get_store()
        known = 0
        while not self._stop_event.is_set():
            count = int(store.add(_CLIENT_COUNT_KEY, 0))
            for client_idx in range(known, count):
                thread = threading.Thread(
                    target=self._serve_client,
                    args=(client_idx,),
                    daemon=True,
                    name=f"DistributedTransport-server-{client_idx}",
                )
                thread.start()
            known = max(known, count)
            time.sleep(0.1)

    def _serve_client(self, client_idx: int) -> None:
        endpoint = _retry_while(
            self._stop_event,
            lambda: _PairEndpoint(
                self._get_store(),
                client_idx,
                rank=0,
                backend=self._backend,
                timeout=self._timeout,
                device=self._device,
            ),
            f"pair rendezvous for client {client_idx}",
        )
        if endpoint is None:
            return
        self._endpoints[client_idx] = endpoint
        self._response_buffers[client_idx] = self._response_spec.clone().to(
            self._device
        )
        request_buffer = self._request_spec.clone().to(self._device)
        recv_leaves = [request_buffer.get(key) for key in self._request_keys]
        header = endpoint.new_header()
        while not self._stop_event.is_set():
            done = _retry_while(
                self._stop_event,
                lambda: endpoint.ctrl_req.recv([header], 1, _TAG_HEADER).wait(),
                f"request header recv for client {client_idx}",
            )
            if done is None and self._stop_event.is_set():
                return
            req_id = int(header[0])
            endpoint.recv(endpoint.data_req, recv_leaves, 1)
            # Clone: the staging buffer is reused for this client's next
            # request, which may be drained before this item is consumed.
            self._ready_queue.put(
                (request_buffer.clone(), (client_idx, req_id), time.monotonic())
            )

    # -- InferenceTransport interface -------------------------------------------

    def wait_for_work(self, timeout: float) -> None:
        """Block until a request has been received or *timeout* elapses."""
        self._ensure_server()
        if self._peeked is not None:
            return
        try:
            self._peeked = self._ready_queue.get(timeout=timeout)
        except queue.Empty:
            return

    def drain(
        self, max_items: int
    ) -> tuple[list[TensorDictBase], list[tuple[int, int]]]:
        """Dequeue up to *max_items* received requests (non-blocking)."""
        items, callbacks, _received_at = self.drain_with_timing(max_items)
        return items, callbacks

    def drain_with_timing(
        self, max_items: int
    ) -> tuple[list[TensorDictBase], list[tuple[int, int]], list[float | None]]:
        """Dequeue requests with server-side arrival timestamps.

        Timestamps are recorded when the receiver thread finished reading
        the payload (clocks are not comparable across nodes, so actor-side
        submission times are not transmitted); queue-wait statistics
        therefore measure server-side queueing only.
        """
        self._ensure_server()
        items: list[TensorDictBase] = []
        callbacks: list[tuple[int, int]] = []
        received_at: list[float | None] = []

        def append(entry) -> None:
            item, callback, timestamp = entry
            items.append(item)
            callbacks.append(callback)
            received_at.append(timestamp)

        if self._peeked is not None:
            append(self._peeked)
            self._peeked = None
        while len(items) < max_items:
            try:
                append(self._ready_queue.get(block=False))
            except queue.Empty:
                break
        return items, callbacks, received_at

    def resolve(self, callback: tuple[int, int], result: TensorDictBase) -> None:
        """Send the result back to the client that submitted the request."""
        client_idx, req_id = callback
        endpoint = self._endpoints[client_idx]
        buffer = self._response_buffers[client_idx]
        buffer.update_(result.select(*self._response_keys, strict=True))
        header = torch.tensor([req_id, _STATUS_OK], dtype=torch.int64)
        endpoint.ctrl_resp.send([header], 1, _TAG_HEADER).wait()
        send_leaves = [buffer.get(key) for key in self._response_keys]
        endpoint.send(endpoint.data_resp, send_leaves, 1)

    def resolve_exception(self, callback: tuple[int, int], exc: BaseException) -> None:
        """Propagate an exception through the store plus a status header."""
        client_idx, req_id = callback
        endpoint = self._endpoints[client_idx]
        try:
            raw = pickle.dumps(exc)
        except Exception:
            raw = pickle.dumps(RuntimeError(repr(exc)))
        self._get_store().set(_exc_key(client_idx, req_id), raw)
        header = torch.tensor([req_id, _STATUS_EXC], dtype=torch.int64)
        endpoint.ctrl_resp.send([header], 1, _TAG_HEADER).wait()

    def close(self) -> None:
        """Stop the server-side listener and receiver threads."""
        if self._stop_event is not None:
            self._stop_event.set()
