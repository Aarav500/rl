.. currentmodule:: torchrl.modules.inference_server

Inference Server
================

.. _ref_inference_server:

The inference server provides auto-batching model serving for RL actors.
Multiple actors submit individual TensorDicts; the server transparently
batches them, runs a single model forward pass, and routes results back.

Core API
--------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    InferenceServer
    InferenceServerConfig
    InferenceDeviceConfig
    ProcessInferenceServer
    InferenceClient
    PolicyClientModule
    InferenceTransport

Transport Backends
------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ThreadingTransport
    SlotTransport
    MPTransport
    SharedMemoryTransport
    DistributedTransport
    RayTransport
    MonarchTransport

Transport Selection
-------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    make_inference_transport
    inference_transport_kind

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    TransportConfig

Usage
-----

The simplest setup uses :class:`ThreadingTransport` for actors that are
threads in the same process:

.. code-block:: python

    from tensordict.nn import TensorDictModule
    from torchrl.modules.inference_server import (
        InferenceServer,
        ThreadingTransport,
    )
    import torch.nn as nn
    import concurrent.futures

    policy = TensorDictModule(
        nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 4)),
        in_keys=["observation"],
        out_keys=["action"],
    )

    transport = ThreadingTransport()
    server = InferenceServer(policy, transport, max_batch_size=32)
    server.start()
    client = server.client()

    # actor threads call client(td) -- batched automatically
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        ...

    server.shutdown()

Shared-memory transport
^^^^^^^^^^^^^^^^^^^^^^^

For cross-process actors with large request payloads (e.g. image
observations), :class:`SharedMemoryTransport` preallocates request and
response slot banks in CPU shared memory and passes only slot indices
through the multiprocessing queues, removing per-request pickling from the
hot path. The caller provides representative request and response
TensorDicts that fix the slot layout (keys, shapes, dtypes); only the
declared keys are transmitted:

.. code-block:: python

    import torch
    from tensordict import TensorDict
    from torchrl.modules.inference_server import (
        InferenceServer,
        SharedMemoryTransport,
    )

    transport = SharedMemoryTransport(
        request_spec=TensorDict({"pixels": torch.zeros(3, 224, 224)}),
        response_spec=TensorDict(
            {
                "action": torch.zeros(7),
                "policy_version": torch.zeros((), dtype=torch.long),
            }
        ),
        num_slots=64,
    )
    # Create clients before spawning env workers
    clients = [transport.client() for _ in range(n_workers)]

    server = InferenceServer(policy, transport, policy_device="cuda:0")
    server.start()

Slots are CPU-only: clients must submit CPU tensors, and the server owns
all device transfers (batches are moved to ``policy_device`` before the
forward pass, results copied back into the CPU response slots).
``num_slots`` bounds the number of concurrently in-flight requests and
provides natural backpressure.

Choosing placement and transport
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TorchRL separates *where a service owner runs* from *how payloads move*:

- ``service_backend`` (and the collector ``backend``/``policy_backend``
  spellings) chooses ownership and placement: thread, process, Ray, Monarch.
- ``transport`` chooses how request/response payloads move: ``"queue"``,
  ``"shared_memory"``, ``"ray"``, ``"monarch"``, or ``"auto"``.
- ``.client()`` hides both choices from consumer code.

The ``transport`` argument of :class:`InferenceServer`,
:class:`ProcessInferenceServer`, and
:class:`~torchrl.collectors.AsyncBatchedCollector` accepts a concrete
:class:`InferenceTransport` instance, a kind string, a
:class:`~torchrl._comm.TransportConfig`, or ``None``/``"auto"``. Strings and
configs are resolved deterministically by :func:`make_inference_transport`;
``"auto"`` picks the fastest safe choice for the topology:

- thread actors with a known slot count: :class:`SlotTransport` (direct
  references, no serialization);
- thread actors otherwise: :class:`ThreadingTransport`;
- process actors with a static TensorDict contract (``request_spec`` and
  ``response_spec`` provided): :class:`SharedMemoryTransport`;
- process actors otherwise: :class:`MPTransport`;
- Ray/Monarch actors: :class:`RayTransport` / :class:`MonarchTransport`.

Placement and payload movement compose independently. In particular, Ray
placement does not force Ray payloads: ``transport="distributed"`` moves
fixed-spec payloads between Ray actors (or any remote processes) through
torch.distributed point-to-point ops -- gloo for CPU tensors, NCCL for
GPU-direct CUDA tensors -- while headers and exceptions ride small gloo
control messages. :class:`DistributedTransport` builds standalone pairwise
process groups rendezvoused through its own
:class:`~torch.distributed.TCPStore`, so the default process group remains
free for training collectives. It is never chosen by ``auto`` (it needs a
static payload contract), so request it explicitly:

.. code-block:: python

    transport = make_inference_transport(
        "distributed",
        request_spec=example_obs,
        response_spec=example_action,
        transport_options={"backend": "nccl"},  # GPU-direct payloads
    )
    clients = [transport.client() for _ in range(n_actors)]  # picklable

Forcing a kind that is incompatible with the topology raises instead of
silently degrading, and the resolved choice is inspectable through the
``transport_kind`` and ``transport_config`` properties:

.. code-block:: python

    from torchrl.collectors import AsyncBatchedCollector

    collector = AsyncBatchedCollector(
        create_env_fn=[make_env] * 8,
        policy_factory=make_policy,
        frames_per_batch=200,
        policy_backend="multiprocessing",
        transport="auto",
        transport_options={
            "request_spec": example_obs,      # single-request TensorDict
            "response_spec": example_action,  # single-response TensorDict
        },
    )
    assert collector.transport_kind == "shared_memory"

Only the worker-facing configuration changes between transports; the env
worker loop and the policy are untouched.

Structured Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Server execution, batching, and device placement are grouped into two
dataclasses instead of loose keyword arguments: :class:`InferenceServerConfig`
collects the execution ``service_backend`` (``"thread"`` or ``"process"``) and the
batching/instrumentation knobs (``max_batch_size``, ``min_batch_size``,
``timeout``, ``collect_stats``, ``stats_window_size``), and
:class:`InferenceDeviceConfig` describes device placement across the
collection pipeline (``policy_device``, ``output_device``, ``env_device``,
``storing_device``). Both :class:`InferenceServer` and
:class:`~torchrl.collectors.AsyncBatchedCollector` accept them through the
``server_config`` and ``device_config`` keyword arguments; a config object is
mutually exclusive with the individual keyword arguments it replaces, and the
config objects are the only way to set the per-role devices and the server
backend on the collector. Servers consume only the
``policy_device``/``output_device`` fields (``env_device`` doubles as an
``output_device`` fallback), while ``env_device`` and ``storing_device``
drive the collector-side transfers:

.. code-block:: python

    from torchrl.collectors import AsyncBatchedCollector
    from torchrl.modules.inference_server import (
        InferenceDeviceConfig,
        InferenceServerConfig,
    )

    collector = AsyncBatchedCollector(
        create_env_fn=[make_env] * 8,
        policy=my_policy,
        frames_per_batch=200,
        server_config=InferenceServerConfig(max_batch_size=8, timeout=0.005),
        device_config=InferenceDeviceConfig(
            policy_device="cuda:0",
            env_device="cpu",
            storing_device="cpu",
        ),
    )

Remote policy module
^^^^^^^^^^^^^^^^^^^^

Use :class:`PolicyClientModule` when an actor or collector expects a regular
TensorDict policy but inference should be served by the policy server:

.. code-block:: python

    remote_policy = PolicyClientModule(
        server,
        in_keys=["observation"],
        out_keys=["action", "policy_version"],
    )

``PolicyClientModule`` accepts a server owner, transport, or existing callable
client. Owners and transports are automatically reduced to their restricted
client before the module is sent to a worker.

    data = remote_policy(data)

The server writes ``policy_version`` by default so asynchronous collectors can
track behavior-policy lag. This is the general *service-stamped metadata*
pattern: any service may stamp its responses with metadata about the state it
served them from, and the data pipeline may enforce freshness constraints on
it. Bounded staleness is enforced by the replay buffer through
:class:`~torchrl.envs.transforms.PolicyAgeFilter`, which drops elements whose
stamped version lags the live version by more than ``max_policy_lag`` --
either at extension time or dynamically at sampling time.

Weight Synchronisation
^^^^^^^^^^^^^^^^^^^^^^

The server integrates with :class:`~torchrl.weight_update.WeightSyncScheme`
to receive updated model weights from a trainer between inference batches:

.. code-block:: python

    from torchrl.weight_update import SharedMemWeightSyncScheme

    weight_sync = SharedMemWeightSyncScheme()
    # Initialise on the trainer (sender) side first
    weight_sync.init_on_sender(model=training_model, ...)

    server = InferenceServer(
        model=inference_model,
        transport=ThreadingTransport(),
        weight_sync=weight_sync,
    )
    server.start()

    # Training loop
    for batch in dataloader:
        loss = loss_fn(training_model(batch))
        loss.backward()
        optimizer.step()
        weight_sync.send(model=training_model)  # pushed to server

Integration with Collectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to use the inference server with RL data collection is
through :class:`~torchrl.collectors.AsyncBatchedCollector`, which
creates the server, transport, and env pool automatically:

.. code-block:: python

    from torchrl.collectors import AsyncBatchedCollector
    from torchrl.envs import GymEnv

    collector = AsyncBatchedCollector(
        create_env_fn=[lambda: GymEnv("CartPole-v1")] * 8,
        policy=my_policy,
        frames_per_batch=200,
        total_frames=10_000,
        max_batch_size=8,
    )

    for data in collector:
        # train on data ...
        pass

    collector.shutdown()
