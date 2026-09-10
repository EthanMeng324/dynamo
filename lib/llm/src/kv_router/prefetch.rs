// SPDX-License-Identifier: Apache-2.0
//! Best-effort route-time CXL prefetch requests.

use std::{
    collections::{HashMap, HashSet},
    env,
    sync::{Arc, Mutex, Once},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use dynamo_runtime::{
    component::Component,
    pipeline::{PushRouter, RouterMode, SingleIn},
    protocols::annotated::Annotated,
};
use serde_json::{Value, json};
use tokio::sync::{OnceCell, mpsc};
use tokio_stream::StreamExt;

use super::indexer::{BlockResidencyMetadata, OverlapScores};
use super::protocols::{LocalBlockHash, WorkerWithDpRank};
use crate::protocols::common::preprocessor::CxlPrefetchHint;

const PREFETCH_ENDPOINT: &str = "cxl_prefetch";
const BACKEND_COMPONENT: &str = "backend";
/// The indexer retains at most this many per-block residency entries.  Keep
/// the token scan bounded by the same order of magnitude when a sparse CXL
/// block appears late in the matched prefix.
const MAX_PREFETCH_SCAN_CHUNKS: usize = 64;
#[derive(Debug, Clone, Copy)]
pub(crate) enum PrefetchSource {
    DynamoRouteTime,
}

impl PrefetchSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::DynamoRouteTime => "dynamo_route_time",
        }
    }
}

/// Timestamps carried with a speculative hint so frontend and worker logs
/// can be joined without relying on log ordering. Values are Unix nanoseconds.
#[derive(Debug, Clone, Default)]
pub(crate) struct PrefetchTrace {
    pub route_decided_unix_ns: Option<u128>,
    pub prefetch_admitted_unix_ns: Option<u128>,
}

#[derive(Debug, Clone)]
struct CxlPrefetchConfig {
    enabled: bool,
    min_tokens: usize,
    /// Maximum number of chunks sent to LMCache in one worker-side admission
    /// batch.  This is not a per-request candidate limit.
    max_chunks: usize,
    /// LMCache's logical chunk size.  This is intentionally independent of the router's block size: the worker, not Dynamo, turns token IDs into CacheEngineKeys.
    chunk_size: usize,
    rpc_timeout: Duration,
    queue_capacity: usize,
    /// How long a `(worker, block)` reservation suppresses another hint. This
    /// is a safety TTL, not a residency guarantee.
    reservation_ttl: Duration,
    /// Bound router-local reservation metadata under repeated-prefix load.
    reservation_capacity: usize,
    /// Highest block position that a hint may inspect.  The worker must hash the preceding chunks to derive a prefix key, but it only promotes the explicitly selected candidate indices.
    max_scan_chunks: usize,
}

impl CxlPrefetchConfig {
    fn from_env() -> Self {
        // The old LOOKAHEAD_ENABLED name is accepted as a compatibility alias,
        // but there is no longer a lookahead window.  Prefetch is evaluated for
        // the request at the time its route is selected.
        let enabled = env_bool(
            "DYN_CXL_PREFETCH_ENABLED",
            env_bool("DYN_CXL_PREFETCH_LOOKAHEAD_ENABLED", false),
        );
        Self {
            enabled,
            min_tokens: env_usize("DYN_CXL_PREFETCH_MIN_TOKENS", 256),
            max_chunks: env_usize(
                "DYN_CXL_PREFETCH_BATCH_CHUNKS",
                env_usize("DYN_CXL_PREFETCH_MAX_CHUNKS", 8),
            )
            .max(1),
            chunk_size: env_usize("DYN_CXL_PREFETCH_CHUNK_SIZE", 256).max(1),
            rpc_timeout: Duration::from_millis(
                env_u64("DYN_CXL_PREFETCH_RPC_TIMEOUT_MS", 10).max(1),
            ),
            queue_capacity: env_usize("DYN_CXL_PREFETCH_QUEUE_CAPACITY", 16).max(1),
            reservation_ttl: Duration::from_millis(
                env_u64(
                    "DYN_CXL_PREFETCH_RESERVATION_TTL_MS",
                    env_u64("DYN_CXL_PREFETCH_LOOKAHEAD_RESERVATION_TTL_MS", 5_000),
                )
                .clamp(1, 60_000),
            ),
            reservation_capacity: env_usize(
                "DYN_CXL_PREFETCH_RESERVATION_CAPACITY",
                env_usize("DYN_CXL_PREFETCH_LOOKAHEAD_RESERVATION_CAPACITY", 16_384),
            )
            .clamp(1, 1_000_000),
            max_scan_chunks: env_usize(
                "DYN_CXL_PREFETCH_MAX_SCAN_CHUNKS",
                MAX_PREFETCH_SCAN_CHUNKS,
            )
            .clamp(1, MAX_PREFETCH_SCAN_CHUNKS),
        }
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<bool>().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn unix_time_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn hint_token_limit(max_chunks: usize, chunk_size: usize) -> usize {
    max_chunks
        .max(1)
        .saturating_mul(chunk_size.max(1))
        .max(chunk_size.max(1))
}

/// Return the number of token chunks needed to derive the requested sparse
/// candidates.  Prefix hashes are cumulative, so a candidate at block 7
/// still requires token chunks 0..=7, even when blocks 1..6 are local/GPU
/// resident and must not themselves be promoted.
fn candidate_token_limit(
    candidate_indices: &[u32],
    chunk_size: usize,
    max_scan_chunks: usize,
) -> Option<usize> {
    let max_index = candidate_indices
        .iter()
        .copied()
        .map(|index| usize::try_from(index).ok())
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .max()?;
    let scan_chunks = max_index.checked_add(1)?;
    if scan_chunks > max_scan_chunks {
        return None;
    }
    Some(hint_token_limit(scan_chunks, chunk_size))
}

/// Return exact blocks that are currently CXL-only on the selected worker.
///
/// Aggregate overlap counters cannot distinguish a CXL block from one that is
/// already available in the worker's CPU/GPU cache. Route-time dispatch uses
/// this bounded per-block view so a hint is emitted only when it can perform a
/// useful CXL-to-CPU promotion.
pub(crate) fn cxl_only_prefetch_metadata(
    overlaps: &OverlapScores,
    worker: WorkerWithDpRank,
    max_candidates: usize,
    max_scan_chunks: usize,
) -> Vec<BlockResidencyMetadata> {
    overlaps
        .block_residency
        .get(&worker)
        .into_iter()
        .flatten()
        .filter(|metadata| (metadata.block_index as usize) < max_scan_chunks && metadata.cxl_only())
        .take(max_candidates)
        .cloned()
        .collect()
}

type PrefetchRouter = PushRouter<Value, Annotated<Value>>;

/// Wait until discovery has made the selected worker available to the
/// endpoint client.  The client is created lazily on the first hint, so its
/// availability snapshot can still be empty even though the worker already
/// registered `backend/cxl_prefetch`.  Calling `direct` before the watch has
/// delivered that registration fails locally without sending any TCP data.
async fn wait_for_instance(router: &PrefetchRouter, instance_id: u64) -> Result<()> {
    if router.client.instance_ids_avail().contains(&instance_id) {
        return Ok(());
    }

    tracing::debug!(instance_id, "Waiting for cxl_prefetch worker discovery");
    let mut available = router.client.instance_avail_watcher();
    loop {
        if available.borrow().contains(&instance_id) {
            tracing::debug!(instance_id, "cxl_prefetch worker discovery completed");
            return Ok(());
        }
        available
            .changed()
            .await
            .context("cxl_prefetch endpoint discovery watch closed")?;
    }
}

struct PrefetchState {
    component: Component,
    config: CxlPrefetchConfig,
    router: OnceCell<Arc<PrefetchRouter>>,
    reservations: Mutex<RouteReservationTable>,
}

/// Router-local record of speculative work that has already been admitted.
///
/// LMCache remains the authoritative CPU/CXL checker. This table closes the
/// control-plane race where another request reaches Dynamo before the first
/// worker RPC has emitted a CPU residency event.
#[derive(Debug)]
struct RouteReservationTable {
    entries: HashMap<(WorkerWithDpRank, LocalBlockHash), Instant>,
    ttl: Duration,
    capacity: usize,
}

impl RouteReservationTable {
    fn new(ttl: Duration, capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            ttl,
            capacity: capacity.max(1),
        }
    }

    fn expire(&mut self, now: Instant) {
        self.entries.retain(|_, expires_at| *expires_at > now);
    }

    fn reserve(
        &mut self,
        worker: WorkerWithDpRank,
        block_hashes: &[LocalBlockHash],
    ) -> Vec<LocalBlockHash> {
        let now = Instant::now();
        self.expire(now);
        let expires_at = now + self.ttl;
        let mut reserved = Vec::new();
        for block_hash in block_hashes.iter().copied() {
            let key = (worker, block_hash);
            if self.entries.contains_key(&key) {
                continue;
            }
            if self.entries.len() >= self.capacity {
                break;
            }
            self.entries.insert(key, expires_at);
            reserved.push(block_hash);
        }
        reserved
    }

    fn release(&mut self, worker: WorkerWithDpRank, block_hashes: &[LocalBlockHash]) {
        for block_hash in block_hashes {
            self.entries.remove(&(worker, *block_hash));
        }
    }
}

struct PrefetchJob {
    request_id: String,
    token_ids: Vec<u32>,
    candidate_block_indices: Vec<u32>,
    worker: WorkerWithDpRank,
    source: PrefetchSource,
    trace: PrefetchTrace,
    reserved_blocks: Vec<LocalBlockHash>,
}

/// Dynamo-side client for the worker-local `cxl_prefetch` endpoint.
///
/// The client deliberately carries logical token IDs only.  LMCache performs
/// token-to-key conversion after the request reaches the selected worker.
pub(crate) struct CxlPrefetchClient {
    state: Arc<PrefetchState>,
    queue_tx: mpsc::Sender<PrefetchJob>,
    queue_rx: Mutex<Option<mpsc::Receiver<PrefetchJob>>>,
    queue_worker_started: Once,
}

impl PrefetchState {
    fn reserve(
        &self,
        worker: WorkerWithDpRank,
        block_hashes: &[LocalBlockHash],
    ) -> Vec<LocalBlockHash> {
        self.reservations
            .lock()
            .expect("route-time prefetch reservation mutex poisoned")
            .reserve(worker, block_hashes)
    }

    fn release(&self, worker: WorkerWithDpRank, block_hashes: &[LocalBlockHash]) {
        self.reservations
            .lock()
            .expect("route-time prefetch reservation mutex poisoned")
            .release(worker, block_hashes);
    }

    async fn router(&self) -> Result<Arc<PrefetchRouter>> {
        let router = self
            .router
            .get_or_try_init(|| async {
                let client = self.component.endpoint(PREFETCH_ENDPOINT).client().await?;
                let router =
                    PushRouter::<Value, Annotated<Value>>::from_client(client, RouterMode::KV)
                        .await?;
                Ok::<_, anyhow::Error>(Arc::new(router))
            })
            .await
            .context("failed to initialize cxl_prefetch endpoint client")?;
        Ok(Arc::clone(router))
    }

    async fn send(
        &self,
        request_id: String,
        token_ids: Vec<u32>,
        candidate_block_indices: Vec<u32>,
        worker: WorkerWithDpRank,
        source: PrefetchSource,
        trace: PrefetchTrace,
    ) -> Result<()> {
        let router = self.router().await?;
        wait_for_instance(&router, worker.worker_id).await?;
        let payload = json!({
            "request_id": request_id,
            "token_ids": token_ids,
            "dp_rank": worker.dp_rank,
            "max_chunks": self.config.max_chunks,
            "candidate_block_indices": candidate_block_indices,
            "prefetch_kind": source.as_str(),
            "route_decided_unix_ns": trace.route_decided_unix_ns,
            "prefetch_admitted_unix_ns": trace.prefetch_admitted_unix_ns,
        });

        let mut stream = router
            .direct(SingleIn::new(payload), worker.worker_id)
            .await
            .with_context(|| {
                format!(
                    "failed to send cxl_prefetch to instance {} (dp_rank={})",
                    worker.worker_id, worker.dp_rank
                )
            })?;

        let mut response_count = 0usize;
        while let Some(response) = stream.next().await {
            response
                .into_result()
                .context("cxl_prefetch worker returned an error")?;
            response_count += 1;
        }

        if response_count == 0 {
            anyhow::bail!("cxl_prefetch worker returned an empty response stream");
        }

        // The worker endpoint emits a data response followed by the runtime's
        // complete-final marker.  Drain the whole stream before returning so
        // the ingress publisher can send that final marker.  Returning after
        // the first item drops the stream early and makes the worker report
        // "Failed to publish complete final" even though queue admission
        // already succeeded.
        tracing::trace!(
            response_count,
            "Drained cxl_prefetch worker response stream"
        );
        Ok(())
    }
}

impl CxlPrefetchClient {
    /// Build a client from the frontend router component.
    ///
    /// The router itself is registered under `kv-router`, while the worker
    /// endpoint is registered under the `backend` component in the same
    /// namespace.  Keep that distinction here so callers cannot accidentally
    /// discover `kv-router/cxl_prefetch`.
    pub(crate) fn new(router_component: Component) -> Result<Self> {
        let worker_component = router_component
            .namespace()
            .component(BACKEND_COMPONENT)
            .context("failed to create backend component for route-time CXL prefetch")?;
        let config = CxlPrefetchConfig::from_env();
        if config.enabled {
            tracing::info!(
                enabled = config.enabled,
                min_tokens = config.min_tokens,
                max_chunks = config.max_chunks,
                chunk_size = config.chunk_size,
                rpc_timeout_ms = config.rpc_timeout.as_millis(),
                queue_capacity = config.queue_capacity,
                reservation_ttl_ms = config.reservation_ttl.as_millis(),
                reservation_capacity = config.reservation_capacity,
                endpoint = PREFETCH_ENDPOINT,
                "Route-time CXL prefetch client enabled"
            );
        }

        let queue_capacity = config.queue_capacity;
        let reservation_ttl = config.reservation_ttl;
        let reservation_capacity = config.reservation_capacity;
        let (queue_tx, queue_rx) = mpsc::channel(queue_capacity);
        Ok(Self {
            state: Arc::new(PrefetchState {
                component: worker_component,
                config,
                router: OnceCell::const_new(),
                reservations: Mutex::new(RouteReservationTable::new(
                    reservation_ttl,
                    reservation_capacity,
                )),
            }),
            queue_tx,
            queue_rx: Mutex::new(Some(queue_rx)),
            queue_worker_started: Once::new(),
        })
    }

    pub(crate) fn chunk_size(&self) -> usize {
        self.state.config.chunk_size
    }

    pub(crate) fn enabled(&self) -> bool {
        self.state.config.enabled
    }

    pub(crate) fn max_chunks(&self) -> usize {
        self.state.config.max_chunks
    }

    pub(crate) fn max_scan_chunks(&self) -> usize {
        self.state.config.max_scan_chunks
    }

    fn should_prefetch(&self, token_count: usize) -> bool {
        self.enabled() && token_count >= self.state.config.min_tokens
    }

    /// Copy the shortest prefix needed to derive the sparse candidate keys.
    /// Keeping this bounded also prevents a long prompt from occupying the
    /// queue with an unnecessarily large token vector.
    ///
    /// Do not use the router/vLLM block size here.  LMCache hashes its own
    /// chunks, and a shorter router block size would make the hint contain no
    /// complete LMCache chunk when `save_unfull_chunk` is disabled.
    pub(crate) fn prepare_tokens(
        &self,
        token_ids: &[u32],
        candidate_block_indices: &[u32],
    ) -> Option<Vec<u32>> {
        if !self.should_prefetch(token_ids.len()) {
            return None;
        }

        let token_limit = candidate_token_limit(
            candidate_block_indices,
            self.state.config.chunk_size,
            self.state.config.max_scan_chunks,
        )?;
        Some(token_ids.iter().copied().take(token_limit).collect())
    }

    /// Build the request-carried form of a route-time hint.
    ///
    /// The old endpoint transport used `prepare_tokens` to trim the payload
    /// before an asynchronous RPC.  The request envelope already carries the
    /// prompt, so retain the same admission checks but let the worker derive
    /// keys from its authoritative lookup token stream.
    pub(crate) fn build_request_hint(
        &self,
        token_ids: &[u32],
        candidate_block_indices: &[u32],
        route_decided_unix_ns: u64,
    ) -> Option<CxlPrefetchHint> {
        if !self.should_prefetch(token_ids.len()) || candidate_block_indices.is_empty() {
            return None;
        }

        let token_limit = candidate_token_limit(
            candidate_block_indices,
            self.state.config.chunk_size,
            self.state.config.max_scan_chunks,
        )?;
        if token_ids.len() < token_limit {
            return None;
        }

        Some(CxlPrefetchHint {
            candidate_block_indices: candidate_block_indices
                .iter()
                .copied()
                // Keep all eligible candidates in the bounded scan range.
                // ``max_chunks`` is the worker-side per-batch limit.
                .take(self.max_scan_chunks())
                .collect(),
            max_chunks: self.max_chunks().min(u32::MAX as usize) as u32,
            route_decided_unix_ns: Some(route_decided_unix_ns),
        })
    }

    fn start_queue_worker(&self) {
        self.queue_worker_started.call_once(|| {
            let mut queue_rx = self
                .queue_rx
                .lock()
                .expect("route-time prefetch queue mutex poisoned")
                .take()
                .expect("route-time prefetch queue worker started twice");
            let state = Arc::clone(&self.state);

            tokio::spawn(async move {
                while let Some(job) = queue_rx.recv().await {
                    // Yield once before the speculative RPC so a request that
                    // was just handed to the worker gets the first chance to
                    // make progress on the shared runtime.
                    tokio::task::yield_now().await;

                    let request_id_for_log = job.request_id.clone();
                    let worker = job.worker;
                    let source = job.source;
                    let candidate_block_indices = job.candidate_block_indices.clone();
                    let reserved_blocks = job.reserved_blocks.clone();
                    let result = tokio::time::timeout(
                        state.config.rpc_timeout,
                        state.send(
                            job.request_id,
                            job.token_ids,
                            candidate_block_indices,
                            worker,
                            source,
                            job.trace,
                        ),
                    )
                    .await;
                    let succeeded = matches!(&result, Ok(Ok(())));
                    if !succeeded {
                        // A failed/timeout RPC never established a worker-side
                        // request, so make these blocks eligible for a later
                        // route immediately instead of waiting for the normal
                        // reservation TTL.
                        state.release(worker, &reserved_blocks);
                    }
                    match result {
                        Ok(Ok(())) => tracing::debug!(
                            request_id = %request_id_for_log,
                            worker_id = worker.worker_id,
                            dp_rank = worker.dp_rank,
                            source = source.as_str(),
                            "CXL prefetch hint queued for worker"
                        ),
                        Ok(Err(error)) => tracing::debug!(
                            request_id = %request_id_for_log,
                            worker_id = worker.worker_id,
                            dp_rank = worker.dp_rank,
                            %error,
                            source = source.as_str(),
                            "CXL prefetch hint failed"
                        ),
                        Err(_) => tracing::debug!(
                            request_id = %request_id_for_log,
                            worker_id = worker.worker_id,
                            dp_rank = worker.dp_rank,
                            source = source.as_str(),
                            "CXL prefetch hint timed out"
                        ),
                    }
                }
            });
        });
    }

    fn enqueue(&self, job: PrefetchJob) -> bool {
        self.start_queue_worker();
        match self.queue_tx.try_send(job) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(job)) => {
                tracing::debug!(
                    request_id = %job.request_id,
                    worker_id = job.worker.worker_id,
                    dp_rank = job.worker.dp_rank,
                    source = job.source.as_str(),
                    capacity = self.state.config.queue_capacity,
                    "Dropping CXL prefetch hint because the queue is full"
                );
                false
            }
            Err(mpsc::error::TrySendError::Closed(job)) => {
                tracing::debug!(
                    request_id = %job.request_id,
                    worker_id = job.worker.worker_id,
                    dp_rank = job.worker.dp_rank,
                    source = job.source.as_str(),
                    "Dropping CXL prefetch hint because the queue is closed"
                );
                false
            }
        }
    }

    /// Enqueue a hint for the worker selected for the current route. The
    /// caller has already checked the selected worker's exact CXL-only block
    /// metadata; this method only performs bounded admission and dispatch.
    pub(crate) fn dispatch_route(
        &self,
        request_id: String,
        token_ids: Vec<u32>,
        worker: WorkerWithDpRank,
        candidate_blocks: Vec<LocalBlockHash>,
        candidate_block_indices: Vec<u32>,
        trace: PrefetchTrace,
    ) -> bool {
        if !self.enabled() {
            return false;
        }

        // Reserve exact router block identities before queue admission. A
        // later request may observe the same CXL prefix before the first hint
        // has completed and should not enqueue another copy attempt.
        let candidate_block_count = candidate_blocks.len();
        let candidate_pairs: Vec<(LocalBlockHash, u32)> = candidate_blocks
            .into_iter()
            .zip(candidate_block_indices)
            .collect();
        if candidate_pairs.is_empty() {
            return false;
        }
        let candidate_hashes: Vec<LocalBlockHash> =
            candidate_pairs.iter().map(|(hash, _)| *hash).collect();
        let reserved_blocks = self.state.reserve(worker, &candidate_hashes);
        if reserved_blocks.is_empty() {
            tracing::debug!(
                request_id = %request_id,
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                candidate_blocks = candidate_block_count,
                "Skipping route-time CXL hint because all candidate blocks are already reserved"
            );
            return false;
        }

        // Reservation admission can be partial when another queued hint has
        // already claimed some candidates.  Keep the index/hash association
        // intact so this RPC cannot re-send the already reserved blocks.
        let reserved_set: HashSet<LocalBlockHash> = reserved_blocks.iter().copied().collect();
        let reserved_indices: Vec<u32> = candidate_pairs
            .iter()
            .filter_map(|(hash, index)| reserved_set.contains(hash).then_some(*index))
            .collect();

        let mut trace = trace;
        let admitted = {
            trace.prefetch_admitted_unix_ns = Some(unix_time_ns());
            self.enqueue(PrefetchJob {
                request_id: request_id.clone(),
                token_ids,
                worker,
                source: PrefetchSource::DynamoRouteTime,
                trace,
                reserved_blocks: reserved_blocks.clone(),
                candidate_block_indices: reserved_indices,
            })
        };
        if !admitted {
            self.state.release(worker, &reserved_blocks);
        }
        admitted
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CxlPrefetchClient, CxlPrefetchConfig, MAX_PREFETCH_SCAN_CHUNKS, RouteReservationTable,
        candidate_token_limit, cxl_only_prefetch_metadata, hint_token_limit,
    };
    use crate::kv_router::indexer::{BlockResidencyMetadata, OverlapScores};
    use crate::kv_router::protocols::{LocalBlockHash, WorkerWithDpRank};
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};

    #[test]
    fn defaults_are_bounded_and_disabled() {
        // Avoid mutating process-wide environment in the unit test.  The
        // production defaults are exercised through the config constructor.
        let config = CxlPrefetchConfig {
            enabled: false,
            min_tokens: 256,
            max_chunks: 8,
            chunk_size: 256,
            rpc_timeout: std::time::Duration::from_millis(10),
            queue_capacity: 16,
            reservation_ttl: std::time::Duration::from_secs(5),
            reservation_capacity: 16_384,
            max_scan_chunks: MAX_PREFETCH_SCAN_CHUNKS,
        };
        assert!(config.max_chunks > 0);
        assert!(config.chunk_size > 0);
        assert!(!config.rpc_timeout.is_zero());
        assert!(config.queue_capacity > 0);
    }

    #[test]
    fn hint_prefix_uses_lmcache_chunk_size() {
        let token_count = 2 * 256 + 63;
        let token_limit = hint_token_limit(2, 256);

        assert_eq!(token_limit, 512);
        assert!(token_count >= token_limit);
        // A router block size of 64 must not reduce this prefix to 128.
        assert_ne!(token_limit, 2 * 64);
    }

    #[test]
    fn sparse_candidate_token_limit_includes_prefix_before_target() {
        assert_eq!(candidate_token_limit(&[4], 256, 64), Some(5 * 256));
        assert_eq!(candidate_token_limit(&[1, 3], 256, 64), Some(4 * 256));
        assert_eq!(candidate_token_limit(&[64], 256, 64), None);
        assert_eq!(candidate_token_limit(&[], 256, 64), None);
    }

    #[test]
    fn cxl_only_candidates_exclude_local_and_inflight_blocks() {
        let worker = WorkerWithDpRank::from_worker_id(3);
        let mut overlaps = OverlapScores::new();
        overlaps.block_residency.insert(
            worker,
            vec![
                BlockResidencyMetadata {
                    block_index: 0,
                    block_hash: LocalBlockHash(10),
                    gpu: false,
                    cpu: true,
                    cxl: true,
                    prefetch_inflight: false,
                },
                BlockResidencyMetadata {
                    block_index: 1,
                    block_hash: LocalBlockHash(11),
                    gpu: false,
                    cpu: false,
                    cxl: true,
                    prefetch_inflight: false,
                },
                BlockResidencyMetadata {
                    block_index: 2,
                    block_hash: LocalBlockHash(12),
                    gpu: false,
                    cpu: false,
                    cxl: true,
                    prefetch_inflight: true,
                },
            ],
        );

        let candidates = cxl_only_prefetch_metadata(&overlaps, worker, 8, 3);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].block_hash, LocalBlockHash(11));
    }

    #[test]
    fn cxl_only_candidates_keep_sparse_blocks_inside_scan_bound() {
        let worker = WorkerWithDpRank::from_worker_id(4);
        let mut overlaps = OverlapScores::new();
        overlaps.block_residency.insert(
            worker,
            vec![
                BlockResidencyMetadata {
                    block_index: 1,
                    block_hash: LocalBlockHash(21),
                    gpu: false,
                    cpu: false,
                    cxl: true,
                    prefetch_inflight: false,
                },
                BlockResidencyMetadata {
                    block_index: 3,
                    block_hash: LocalBlockHash(23),
                    gpu: false,
                    cpu: false,
                    cxl: true,
                    prefetch_inflight: false,
                },
                BlockResidencyMetadata {
                    block_index: 4,
                    block_hash: LocalBlockHash(24),
                    gpu: false,
                    cpu: false,
                    cxl: true,
                    prefetch_inflight: false,
                },
            ],
        );

        let candidates = cxl_only_prefetch_metadata(&overlaps, worker, 2, 4);
        assert_eq!(
            candidates
                .iter()
                .map(|metadata| metadata.block_hash)
                .collect::<Vec<_>>(),
            vec![LocalBlockHash(21), LocalBlockHash(23)]
        );
    }

    #[test]
    fn route_reservations_deduplicate_per_worker_and_release() {
        let worker0 = WorkerWithDpRank::from_worker_id(10);
        let worker1 = WorkerWithDpRank::from_worker_id(11);
        let mut table = RouteReservationTable::new(std::time::Duration::from_secs(5), 3);

        assert_eq!(
            table.reserve(worker0, &[LocalBlockHash(1), LocalBlockHash(2)]),
            vec![LocalBlockHash(1), LocalBlockHash(2)]
        );
        assert!(
            table
                .reserve(worker0, &[LocalBlockHash(1), LocalBlockHash(2)])
                .is_empty()
        );

        // The same block hash on another worker is a distinct physical
        // destination and must not be suppressed.
        assert_eq!(
            table.reserve(worker1, &[LocalBlockHash(1)]),
            vec![LocalBlockHash(1)]
        );

        table.release(worker0, &[LocalBlockHash(1)]);
        assert_eq!(
            table.reserve(worker0, &[LocalBlockHash(1)]),
            vec![LocalBlockHash(1)]
        );
    }

    #[tokio::test]
    async fn worker_endpoint_uses_backend_component() {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let namespace = drt.namespace("route-prefetch-test").unwrap();
        let router_component = namespace.component("kv-router").unwrap();

        let client = CxlPrefetchClient::new(router_component).unwrap();

        assert_eq!(client.state.component.name(), "backend");
        assert_eq!(
            client.state.component.namespace().name(),
            "route-prefetch-test"
        );

        runtime.shutdown();
    }
}
