#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.frontend [args]`
#
# Start a frontend node. This runs:
# - OpenAI HTTP server.
# - Auto-discovery: Watches etcd for engine/worker registration (via `register_llm`).
# - Pre-processor: Prompt templating and tokenization.
# - Router, defaulting to round-robin. Use --router-mode to switch (round-robin, random, kv).
#
# Pass `--interactive` or `-i` for text chat instead of HTTP server.
#
# For TLS:
# - python -m dynamo.frontend --http-port 8443 --tls-cert-path cert.pem --tls-key-path key.pem
#

import argparse
import asyncio
import logging
import os
import pathlib
import signal

import uvloop

from dynamo.common.config_dump import dump_config
from dynamo.common.config_dump.config_dumper import add_config_dump_args
from dynamo.llm import (
    EngineType,
    EntrypointArgs,
    KvRouterConfig,
    ModelDeploymentCard,
    PythonAsyncEngine,
    RouterConfig,
    RouterMode,
    make_engine,
    run_input,
)
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

from . import __version__

DYN_NAMESPACE_ENV_VAR = "DYN_NAMESPACE"
CUSTOM_BACKEND_METRICS_POLLING_INTERVAL_ENV_VAR = (
    "CUSTOM_BACKEND_METRICS_POLLING_INTERVAL"
)
CUSTOM_BACKEND_ENDPOINT_ENV_VAR = "CUSTOM_BACKEND_ENDPOINT"

configure_dynamo_logging()
logger = logging.getLogger(__name__)


async def _dummy_generator(request):
    """Minimal generator that yields nothing. Work in progress."""
    return
    yield  # Makes this an async generator


async def engine_factory(mdc: ModelDeploymentCard) -> PythonAsyncEngine:
    """
    Called by Rust when a model is discovered.
    """
    loop = asyncio.get_running_loop()
    logger.info(f"Engine_factory called with MDC: {mdc.to_json_str()[:100]}...")
    return PythonAsyncEngine(_dummy_generator, loop)


def validate_model_name(value):
    """Validate that model-name is a non-empty string."""
    if not value or not isinstance(value, str) or len(value.strip()) == 0:
        raise argparse.ArgumentTypeError(
            f"model-name must be a non-empty string, got: {value}"
        )
    return value.strip()


def validate_model_path(value):
    """Validate that model-path is a valid directory on disk."""
    if not os.path.isdir(value):
        raise argparse.ArgumentTypeError(
            f"model-path must be a valid directory on disk, got: {value}"
        )
    return value


def parse_args():
    """Parse command-line arguments for the Dynamo frontend.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Dynamo Frontend: HTTP+Pre-processor+Router",
        formatter_class=argparse.RawTextHelpFormatter,  # To preserve multi-line help formatting
    )
    parser.add_argument(
        "--version", action="version", version=f"Dynamo Frontend {__version__}"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive text chat"
    )
    parser.add_argument(
        "--kv-cache-block-size",
        type=int,
        default=os.environ.get("DYN_KV_CACHE_BLOCK_SIZE"),
        help="KV cache block size (u32). Can be set via DYN_KV_CACHE_BLOCK_SIZE env var.",
    )
    parser.add_argument(
        "--http-host",
        type=str,
        default=os.environ.get("DYN_HTTP_HOST", "0.0.0.0"),
        help="HTTP host for the engine (str). Can be set via DYN_HTTP_HOST env var.",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=int(os.environ.get("DYN_HTTP_PORT", "8000")),
        help="HTTP port for the engine (u16). Can be set via DYN_HTTP_PORT env var.",
    )
    parser.add_argument(
        "--tls-cert-path",
        type=pathlib.Path,
        default=None,
        help="TLS certificate path, PEM format.",
    )
    parser.add_argument(
        "--tls-key-path",
        type=pathlib.Path,
        default=None,
        help="TLS certificate key path, PEM format.",
    )
    parser.add_argument(
        "--router-mode",
        type=str,
        choices=["round-robin", "random", "kv", "kv-strata"],
        default=os.environ.get("DYN_ROUTER_MODE", "round-robin"),
        help="How to route the request. 'kv' = original Dynamo logic (GPU cache hit only). 'kv-strata' = consider cache hits from all memory tiers (GPU, CPU, KVBM). Can be set via DYN_ROUTER_MODE env var.",
    )
    parser.add_argument(
        "--kv-overlap-score-weight",
        type=float,
        default=float(os.environ.get("DYN_KV_OVERLAP_SCORE_WEIGHT", "1.0")),
        help="KV Router: Weight for overlap score in worker selection. Higher values prioritize KV cache reuse.",
    )
    parser.add_argument(
        "--router-temperature",
        type=float,
        default=float(os.environ.get("DYN_ROUTER_TEMPERATURE", "0.0")),
        help="KV Router: Temperature for worker sampling via softmax. Higher values promote more randomness, and 0 fallbacks to deterministic.",
    )
    parser.add_argument(
        "--strata-cpu-overlap-weight",
        type=float,
        default=float(os.environ.get("DYN_STRATA_CPU_OVERLAP_WEIGHT", "0.9")),
        help="kv-strata only: weight (<=1.0) applied to CPU cache hits relative to GPU hits when scoring workers. 1.0 counts CPU hits equal to GPU hits; smaller values discount CPU hits to reflect CPU->GPU transfer cost. Ignored in 'kv' mode.",
    )
    parser.add_argument(
        "--strata-cxl-overlap-weight",
        type=float,
        default=float(os.environ.get("DYN_STRATA_CXL_OVERLAP_WEIGHT", "0.8")),
        help="kv-strata only: weight (<=1.0) applied to CXL cache hits relative to GPU hits when scoring workers. Typically lower than --strata-cpu-overlap-weight. Ignored in 'kv' mode.",
    )
    parser.add_argument(
        "--strata-prefetch-overlap-weight",
        type=float,
        default=float(os.environ.get("DYN_STRATA_PREFETCH_OVERLAP_WEIGHT", "0.5")),
        help="kv-strata only: weight for blocks whose CXL-to-CPU prefetch is inflight.",
    )
    parser.add_argument(
        "--strata-load-aware-shared-cpu",
        action=argparse.BooleanOptionalAction,
        default=(
            os.environ.get("DYN_STRATA_LOAD_AWARE_SHARED_CPU", "false").lower()
            == "true"
        ),
        help="kv-strata only: when a block exists in both local CPU and shared CXL, gradually relax the local CPU affinity as that worker's decode and prefill load rises. Disabled by default.",
    )
    parser.add_argument(
        "--strata-randomize-ties",
        action=argparse.BooleanOptionalAction,
        default=(
            os.environ.get("DYN_STRATA_RANDOMIZE_TIES", "false").lower() == "true"
        ),
        help="kv-strata only: randomly choose among workers with the same minimum route cost instead of using a deterministic tree-size tie break. Disabled by default.",
    )
    parser.add_argument(
        "--background-offload",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("DYN_BACKGROUND_OFFLOAD_ENABLED", "false").lower()
        == "true",
        help="Enable the default-off periodic CPU-to-CXL offload planner.",
    )
    parser.add_argument(
        "--background-offload-dry-run",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("DYN_BACKGROUND_OFFLOAD_DRY_RUN", "true").lower()
        == "true",
        help="Log periodic offload candidates without sending LMCache commands.",
    )
    parser.add_argument(
        "--background-offload-interval-ms",
        type=int,
        default=int(os.environ.get("DYN_BACKGROUND_OFFLOAD_INTERVAL_MS", "5000")),
    )
    parser.add_argument(
        "--background-offload-window-ms",
        type=int,
        default=int(os.environ.get("DYN_BACKGROUND_OFFLOAD_WINDOW_MS", "10000")),
    )
    parser.add_argument(
        "--background-offload-hot-request-threshold",
        type=int,
        default=int(
            os.environ.get("DYN_BACKGROUND_OFFLOAD_HOT_REQUEST_THRESHOLD", "5")
        ),
    )
    parser.add_argument(
        "--background-offload-owner-load-threshold",
        type=int,
        default=int(
            os.environ.get("DYN_BACKGROUND_OFFLOAD_OWNER_LOAD_THRESHOLD", "10")
        ),
    )
    parser.add_argument(
        "--background-offload-top-k",
        type=int,
        default=int(os.environ.get("DYN_BACKGROUND_OFFLOAD_TOP_K", "100")),
    )
    parser.add_argument(
        "--background-offload-max-chunks",
        type=int,
        default=int(os.environ.get("DYN_BACKGROUND_OFFLOAD_MAX_CHUNKS", "8")),
    )
    parser.add_argument(
        "--background-offload-max-inflight",
        type=int,
        default=int(os.environ.get("DYN_BACKGROUND_OFFLOAD_MAX_INFLIGHT", "1")),
        help="Legacy setting retained for compatibility; automatic offload is always single-concurrent.",
    )
    parser.add_argument(
        "--background-offload-max-active-decode-blocks",
        type=int,
        default=int(
            os.environ.get("DYN_BACKGROUND_OFFLOAD_MAX_ACTIVE_DECODE_BLOCKS", "16")
        ),
        help="Admission ceiling for aggregate active decode blocks before a background offload may start.",
    )
    parser.add_argument(
        "--background-offload-max-pending-prefill-tokens",
        type=int,
        default=int(
            os.environ.get("DYN_BACKGROUND_OFFLOAD_MAX_PENDING_PREFILL_TOKENS", "0")
        ),
        help="Admission ceiling for aggregate pending prefill tokens before a background offload may start.",
    )
    parser.add_argument(
        "--background-offload-bytes-per-sec",
        type=int,
        default=int(
            os.environ.get("DYN_BACKGROUND_OFFLOAD_BYTES_PER_SEC", str(64 * 1024 * 1024))
        ),
        help="Sustained background CPU-to-CXL write budget in bytes/sec; zero disables the limiter.",
    )
    parser.add_argument(
        "--background-offload-chunk-bytes",
        type=int,
        default=int(
            os.environ.get("DYN_BACKGROUND_OFFLOAD_CHUNK_BYTES", str(1024 * 1024))
        ),
        help="Conservative bytes-per-KV-chunk estimate used by the offload byte limiter.",
    )
    parser.add_argument(
        "--background-offload-cooldown-secs",
        type=int,
        default=int(os.environ.get("DYN_BACKGROUND_OFFLOAD_COOLDOWN_SECS", "30")),
    )
    parser.add_argument(
        "--lmcache-controller-endpoint",
        default=os.environ.get("DYN_LMCACHE_CONTROLLER_ENDPOINT"),
        help="LMCache controller ZMQ REQ/REP endpoint, e.g. tcp://controller:9000.",
    )
    parser.add_argument(
        "--background-offload-instance-map",
        default=os.environ.get("DYN_BACKGROUND_OFFLOAD_INSTANCE_MAP", ""),
        help="Static worker-to-LMCache map, e.g. '0=instance-0,1=instance-1'.",
    )
    parser.add_argument(
        "--background-offload-instance-template",
        default=os.environ.get("DYN_BACKGROUND_OFFLOAD_INSTANCE_TEMPLATE"),
        help="LMCache instance template with {worker_id} and optional {dp_rank}.",
    )
    parser.add_argument(
        "--kv-events",
        action=argparse.BooleanOptionalAction,
        dest="use_kv_events",
        default=(
            os.environ.get("DYN_KV_EVENTS", "true").lower() == "true"
        ),  # default is true
        help="KV Router: Enable/disable KV events. Use --kv-events to enable (default, router receives cache state events from workers) or --no-kv-events to disable (router predicts cache state based on routing decisions).",
    )
    parser.add_argument(
        "--router-ttl",
        type=float,
        default=float(os.environ.get("DYN_ROUTER_TTL", "120.0")),
        help="KV Router: Time-to-live in seconds for blocks when KV events are disabled. Only used when --no-kv-events is set. Can be set via DYN_ROUTER_TTL env var (default: 120.0).",
    )
    parser.add_argument(
        "--router-max-tree-size",
        type=int,
        default=int(os.environ.get("DYN_ROUTER_MAX_TREE_SIZE", str(2**20))),
        help="KV Router: Maximum tree size before pruning when KV events are disabled. Only used when --no-kv-events is set. Can be set via DYN_ROUTER_MAX_TREE_SIZE env var (default: 1048576, which is 2^20).",
    )
    parser.add_argument(
        "--router-prune-target-ratio",
        type=float,
        default=float(os.environ.get("DYN_ROUTER_PRUNE_TARGET_RATIO", "0.8")),
        help="KV Router: Target size ratio after pruning when KV events are disabled. Only used when --no-kv-events is set. Can be set via DYN_ROUTER_PRUNE_TARGET_RATIO env var (default: 0.8).",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=os.environ.get(DYN_NAMESPACE_ENV_VAR),
        help="Dynamo namespace for model discovery scoping. If specified, models will only be discovered from this namespace. If not specified, discovers models from all namespaces (global discovery).",
    )
    parser.add_argument(
        "--router-replica-sync",
        action="store_true",
        default=False,
        help="KV Router: Enable replica synchronization across multiple router instances. When true, routers will publish and subscribe to events to maintain consistent state.",
    )
    parser.add_argument(
        "--router-snapshot-threshold",
        type=int,
        default=1000000,
        help="KV Router: Number of messages in stream before triggering a snapshot. Defaults to 1000000.",
    )
    parser.add_argument(
        "--router-reset-states",
        action="store_true",
        dest="router_reset_states",
        default=False,
        help="KV Router: Reset router state on startup, purging stream and object store. By default, states are persisted. WARNING: This can affect existing router replicas.",
    )
    parser.add_argument(
        "--no-track-active-blocks",
        action="store_false",
        dest="router_track_active_blocks",
        default=True,
        help="KV Router: Disable tracking of active blocks (blocks being used for ongoing generation). By default, active blocks are tracked for load balancing.",
    )
    parser.add_argument(
        "--no-assume-kv-reuse",
        action="store_false",
        dest="router_assume_kv_reuse",
        default=True,
        help="KV Router: When tracking active blocks, do not assume KV cache reuse (generate random hashes instead of computing actual block hashes). Useful when KV cache reuse is not expected. By default, KV cache reuse is assumed.",
    )
    parser.add_argument(
        "--track-output-blocks",
        action="store_true",
        dest="router_track_output_blocks",
        default=False,
        help="KV Router: Track output blocks during generation. When enabled, the router adds placeholder blocks as tokens are generated and applies fractional decay based on progress toward expected_output_tokens. By default, output blocks are not tracked.",
    )
    parser.add_argument(
        "--enforce-disagg",
        action="store_true",
        default=False,
        help="Enforce disaggregated prefill-decode. When set, unactivated prefill router will return an error instead of falling back to decode-only mode.",
    )
    parser.add_argument(
        "--active-decode-blocks-threshold",
        type=float,
        default=None,
        help="Threshold percentage (0.0-1.0) for determining when a worker is considered busy based on KV cache block utilization. If not set, blocks-based busy detection is disabled.",
    )
    parser.add_argument(
        "--active-prefill-tokens-threshold",
        type=int,
        default=None,
        help="Literal token count threshold for determining when a worker is considered busy based on prefill token utilization. When active prefill tokens exceed this threshold, the worker is marked as busy. If not set, tokens-based busy detection is disabled.",
    )
    parser.add_argument(
        "--model-name",
        type=validate_model_name,
        help="Model name as a string (e.g., 'Llama-3.2-1B-Instruct')",
    )
    parser.add_argument(
        "--model-path",
        type=validate_model_path,
        help="Path to model directory on disk (e.g., /tmp/model_cache/llama3.2_1B/)",
    )
    parser.add_argument(
        "--metrics-prefix",
        type=str,
        default=None,
        help="Prefix for Dynamo frontend metrics. If unset, uses DYN_METRICS_PREFIX env var or 'dynamo_frontend'.",
    )
    parser.add_argument(
        "--kserve-grpc-server",
        action="store_true",
        default=False,
        help="Start KServe gRPC server.",
    )
    parser.add_argument(
        "--grpc-metrics-port",
        type=int,
        default=8788,
        help="HTTP metrics port for gRPC service (u16). Only used with --kserve-grpc-server. Defaults to 8788.",
    )
    add_config_dump_args(parser)
    parser.add_argument(
        "--custom-backend-metrics-endpoint",
        type=str,
        default=os.environ.get(
            CUSTOM_BACKEND_ENDPOINT_ENV_VAR, "nim.backend.runtime_stats"
        ),
        help=f"Custom backend endpoint to poll for metrics in format 'namespace.component.endpoint' (default: 'nim.backend.runtime_stats'). Required if --custom-backend-metrics-polling-interval is specified. All metrics will be prefixed with 'dynamo_component_' in Prometheus. Can be set via {CUSTOM_BACKEND_ENDPOINT_ENV_VAR} env var.",
    )
    parser.add_argument(
        "--custom-backend-metrics-polling-interval",
        type=float,
        default=float(
            os.environ.get(CUSTOM_BACKEND_METRICS_POLLING_INTERVAL_ENV_VAR, "0")
        ),
        help=f"Interval in seconds for polling custom backend metrics. Set to > 0 to enable polling (default: 0=disabled, suggested: 9.2s which is less than typical Prometheus scrape interval). Can be set via {CUSTOM_BACKEND_METRICS_POLLING_INTERVAL_ENV_VAR} env var.",
    )
    parser.add_argument(
        "--store-kv",
        type=str,
        choices=["etcd", "file", "mem"],
        default=os.environ.get("DYN_STORE_KV", "etcd"),
        help="Which key-value backend to use: etcd, mem, file. Etcd uses the ETCD_* env vars (e.g. ETCD_ENDPOINTS) for connection details. File uses root dir from env var DYN_FILE_KV or defaults to $TMPDIR/dynamo_store_kv.",
    )
    parser.add_argument(
        "--request-plane",
        type=str,
        choices=["nats", "http", "tcp"],
        default=os.environ.get("DYN_REQUEST_PLANE", "tcp"),
        help="Determines how requests are distributed from routers to workers. 'tcp' is fastest [nats|http|tcp]",
    )
    parser.add_argument(
        "--event-plane",
        type=str,
        choices=["nats", "zmq"],
        default=os.environ.get("DYN_EVENT_PLANE", "nats"),
        help="Determines how events are published [nats|zmq]",
    )
    parser.add_argument(
        "--exp-python-factory",
        action="store_true",
        default=False,
        help="[EXPERIMENTAL] Enable Python-based engine factory. When set, engines will be created via a Python callback instead of the default Rust pipeline.",
    )

    flags = parser.parse_args()

    if bool(flags.tls_cert_path) ^ bool(flags.tls_key_path):  # ^ is XOR
        parser.error("--tls-cert-path and --tls-key-path must be provided together")
    if flags.custom_backend_metrics_polling_interval < 0:
        parser.error(
            "--custom-backend-metrics-polling-interval must be >= 0 (0=disabled)"
        )

    return flags


async def async_main():
    """Main async entry point for the Dynamo frontend.

    Initializes the distributed runtime, configures routing, and starts
    the HTTP server or interactive mode based on command-line arguments.
    """
    # The system status server port is a worker concern.
    #
    # Serve tests set DYN_SYSTEM_PORT for the worker, but aggregated launch scripts
    # start `dynamo.frontend` first. If the frontend inherits DYN_SYSTEM_PORT, it can
    # bind that port before the worker, causing port conflicts and/or scraping the
    # wrong metrics endpoint.
    os.environ.pop("DYN_SYSTEM_PORT", None)
    flags = parse_args()
    dump_config(flags.dump_config_to, flags)
    os.environ["DYN_EVENT_PLANE"] = flags.event_plane
    # Warn if DYN_SYSTEM_PORT is set (frontend doesn't use system metrics server)
    if os.environ.get("DYN_SYSTEM_PORT"):
        logger.warning(
            "=" * 80 + "\n"
            "WARNING: DYN_SYSTEM_PORT is set but NOT used by the frontend!\n"
            "The frontend does not expose a system metrics server.\n"
            "Only backend workers should set DYN_SYSTEM_PORT.\n"
            "Use --http-port to configure the frontend HTTP API port.\n" + "=" * 80
        )

    # Configure Dynamo frontend HTTP service metrics prefix
    if flags.metrics_prefix is not None:
        prefix = flags.metrics_prefix.strip()
        if prefix:
            os.environ["DYN_METRICS_PREFIX"] = flags.metrics_prefix

    background_offload_env = {
        "DYN_BACKGROUND_OFFLOAD_ENABLED": flags.background_offload,
        "DYN_BACKGROUND_OFFLOAD_DRY_RUN": flags.background_offload_dry_run,
        "DYN_BACKGROUND_OFFLOAD_INTERVAL_MS": flags.background_offload_interval_ms,
        "DYN_BACKGROUND_OFFLOAD_WINDOW_MS": flags.background_offload_window_ms,
        "DYN_BACKGROUND_OFFLOAD_HOT_REQUEST_THRESHOLD": flags.background_offload_hot_request_threshold,
        "DYN_BACKGROUND_OFFLOAD_OWNER_LOAD_THRESHOLD": flags.background_offload_owner_load_threshold,
        "DYN_BACKGROUND_OFFLOAD_TOP_K": flags.background_offload_top_k,
        "DYN_BACKGROUND_OFFLOAD_MAX_CHUNKS": flags.background_offload_max_chunks,
        "DYN_BACKGROUND_OFFLOAD_MAX_INFLIGHT": flags.background_offload_max_inflight,
        "DYN_BACKGROUND_OFFLOAD_MAX_ACTIVE_DECODE_BLOCKS": flags.background_offload_max_active_decode_blocks,
        "DYN_BACKGROUND_OFFLOAD_MAX_PENDING_PREFILL_TOKENS": flags.background_offload_max_pending_prefill_tokens,
        "DYN_BACKGROUND_OFFLOAD_BYTES_PER_SEC": flags.background_offload_bytes_per_sec,
        "DYN_BACKGROUND_OFFLOAD_CHUNK_BYTES": flags.background_offload_chunk_bytes,
        "DYN_BACKGROUND_OFFLOAD_COOLDOWN_SECS": flags.background_offload_cooldown_secs,
        "DYN_LMCACHE_CONTROLLER_ENDPOINT": flags.lmcache_controller_endpoint,
        "DYN_BACKGROUND_OFFLOAD_INSTANCE_MAP": flags.background_offload_instance_map,
        "DYN_BACKGROUND_OFFLOAD_INSTANCE_TEMPLATE": flags.background_offload_instance_template,
    }
    for name, value in background_offload_env.items():
        if value is None:
            os.environ.pop(name, None)
        elif isinstance(value, bool):
            os.environ[name] = str(value).lower()
        else:
            os.environ[name] = str(value)

    # NATS is needed when:
    # 1. Request plane is NATS, OR
    # 2. Event plane is NATS AND KV router mode (kv or kv-strata) AND (KV events OR replica sync enabled)
    enable_nats = flags.request_plane == "nats" or (
        flags.event_plane == "nats"
        and flags.router_mode in ("kv", "kv-strata")
        and (flags.use_kv_events or flags.router_replica_sync)
    )

    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, flags.store_kv, flags.request_plane, enable_nats)

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    if flags.router_mode in ("kv", "kv-strata"):
        router_mode = RouterMode.KV
        use_strata_routing = flags.router_mode == "kv-strata"
        kv_router_config = KvRouterConfig(
            overlap_score_weight=flags.kv_overlap_score_weight,
            router_temperature=flags.router_temperature,
            use_kv_events=flags.use_kv_events,
            router_replica_sync=flags.router_replica_sync,
            router_track_active_blocks=flags.router_track_active_blocks,
            router_track_output_blocks=flags.router_track_output_blocks,
            router_assume_kv_reuse=flags.router_assume_kv_reuse,
            router_snapshot_threshold=flags.router_snapshot_threshold,
            router_reset_states=flags.router_reset_states,
            router_ttl_secs=flags.router_ttl,
            router_max_tree_size=flags.router_max_tree_size,
            router_prune_target_ratio=flags.router_prune_target_ratio,
            use_strata_routing=use_strata_routing,
            strata_cpu_overlap_weight=flags.strata_cpu_overlap_weight,
            strata_cxl_overlap_weight=flags.strata_cxl_overlap_weight,
            strata_prefetch_overlap_weight=flags.strata_prefetch_overlap_weight,
            strata_load_aware_shared_cpu=flags.strata_load_aware_shared_cpu,
            strata_randomize_ties=flags.strata_randomize_ties,
            enable_background_offload=flags.background_offload,
            background_offload_dry_run=flags.background_offload_dry_run,
            offload_interval_ms=flags.background_offload_interval_ms,
            offload_window_ms=flags.background_offload_window_ms,
            offload_hot_request_threshold=flags.background_offload_hot_request_threshold,
            offload_owner_load_threshold=flags.background_offload_owner_load_threshold,
            offload_top_k=flags.background_offload_top_k,
            offload_max_chunks=flags.background_offload_max_chunks,
            offload_max_inflight=flags.background_offload_max_inflight,
            offload_cooldown_secs=flags.background_offload_cooldown_secs,
        )
    elif flags.router_mode == "random":
        router_mode = RouterMode.Random
        kv_router_config = None
    else:
        router_mode = RouterMode.RoundRobin
        kv_router_config = None

    kwargs = {
        "http_host": flags.http_host,
        "http_port": flags.http_port,
        "kv_cache_block_size": flags.kv_cache_block_size,
        "router_config": RouterConfig(
            router_mode,
            kv_router_config,
            active_decode_blocks_threshold=flags.active_decode_blocks_threshold,
            active_prefill_tokens_threshold=flags.active_prefill_tokens_threshold,
            enforce_disagg=flags.enforce_disagg,
        ),
    }

    if flags.model_name:
        kwargs["model_name"] = flags.model_name
    if flags.model_path:
        kwargs["model_path"] = flags.model_path
    if flags.tls_cert_path:
        kwargs["tls_cert_path"] = flags.tls_cert_path
    if flags.tls_key_path:
        kwargs["tls_key_path"] = flags.tls_key_path
    if flags.namespace:
        kwargs["namespace"] = flags.namespace
    if flags.kserve_grpc_server and flags.grpc_metrics_port:
        kwargs["http_metrics_port"] = flags.grpc_metrics_port
    if flags.custom_backend_metrics_endpoint:
        kwargs[
            "custom_backend_metrics_endpoint"
        ] = flags.custom_backend_metrics_endpoint
    if flags.custom_backend_metrics_polling_interval:
        kwargs[
            "custom_backend_metrics_polling_interval"
        ] = flags.custom_backend_metrics_polling_interval

    if flags.exp_python_factory:
        kwargs["engine_factory"] = engine_factory

    e = EntrypointArgs(EngineType.Dynamic, **kwargs)
    engine = await make_engine(runtime, e)

    try:
        if flags.interactive:
            await run_input(runtime, "text", engine)
        elif flags.kserve_grpc_server:
            await run_input(runtime, "grpc", engine)
        else:
            await run_input(runtime, "http", engine)
    except asyncio.exceptions.CancelledError:
        pass


async def graceful_shutdown(runtime):
    """Handle graceful shutdown of the distributed runtime.

    Args:
        runtime: The DistributedRuntime instance to shut down.
    """
    runtime.shutdown()


def main():
    """Entry point for the Dynamo frontend CLI."""
    uvloop.run(async_main())


if __name__ == "__main__":
    main()
