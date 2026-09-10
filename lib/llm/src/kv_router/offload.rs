// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Periodic, router-owned CPU-to-CXL offload planning.
//!
//! Request handling only records observations.  A bounded background task
//! wakes every few seconds, ranks repeated prefixes by CPU-locality skew, and
//! sends a MessagePack
//! `OffloadMsg` over LMCache's existing ZMQ REQ/REP controller endpoint.  The
//! planner deliberately does not mutate Dynamo residency: only the normal CXL
//! `BlockStored` event can make a copy routable.

use std::{
    collections::HashMap,
    env,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use bytes::Bytes;
use rmp_serde::{from_slice, to_vec_named};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tokio_util::sync::CancellationToken;
use zeromq::{ReqSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use super::{
    KvRouterConfig, compute_block_hash_for_seq,
    indexer::{BlockResidencyMetadata, OverlapScores},
    protocols::WorkerWithDpRank,
};

#[derive(Debug, Clone)]
struct PlannerConfig {
    enabled: bool,
    dry_run: bool,
    interval: Duration,
    window: Duration,
    hot_threshold: u32,
    skew_threshold: f64,
    max_offloads_per_round: usize,
    max_chunks: usize,
    /// Bound concurrent background operations so a planner round cannot
    /// create an unbounded burst of CXL traffic.
    max_inflight: usize,
    /// Do not admit background copies while serving pressure is above these
    /// thresholds.  The values are snapshots supplied by the scheduler.
    max_serving_decode_blocks: u64,
    max_serving_prefill_tokens: u64,
    /// Sustained CXL write budget.  Zero disables the limiter for backwards
    /// compatibility; the frontend default is finite and conservative.
    bytes_per_sec: u64,
    /// Conservative estimate used to reserve a bounded per-operation budget.
    chunk_bytes: u64,
    cooldown: Duration,
    endpoint: Option<String>,
    instance_template: Option<String>,
    instance_map: HashMap<u64, String>,
    timeout: Duration,
}

impl PlannerConfig {
    fn from_config(config: KvRouterConfig) -> Self {
        let legacy_max_offloads_per_round =
            env_usize("DYN_BACKGROUND_OFFLOAD_TOP_K", config.offload_top_k);
        let max_offloads_per_round = env_usize(
            "DYN_BACKGROUND_OFFLOAD_MAX_PER_ROUND",
            // TOP_K is the old public knob. Keep it as a fallback so existing
            // deployments can adopt the new selector without a flag change.
            legacy_max_offloads_per_round,
        )
        .max(1);
        let max_inflight = env_usize(
            "DYN_BACKGROUND_OFFLOAD_MAX_INFLIGHT",
            config.offload_max_inflight,
        )
        .max(1)
        .min(max_offloads_per_round);
        Self {
            enabled: env_bool(
                "DYN_BACKGROUND_OFFLOAD_ENABLED",
                config.enable_background_offload,
            ),
            dry_run: env_bool(
                "DYN_BACKGROUND_OFFLOAD_DRY_RUN",
                config.background_offload_dry_run,
            ),
            interval: Duration::from_millis(
                env_u64(
                    "DYN_BACKGROUND_OFFLOAD_INTERVAL_MS",
                    config.offload_interval_ms,
                )
                .max(1),
            ),
            window: Duration::from_millis(
                env_u64("DYN_BACKGROUND_OFFLOAD_WINDOW_MS", config.offload_window_ms).max(1),
            ),
            hot_threshold: env_u64(
                "DYN_BACKGROUND_OFFLOAD_HOT_REQUEST_THRESHOLD",
                config.offload_hot_request_threshold as u64,
            )
            .max(1) as u32,
            skew_threshold: env_f64("DYN_BACKGROUND_OFFLOAD_SKEW_THRESHOLD", 0.5).clamp(0.0, 1.0),
            max_offloads_per_round,
            max_chunks: env_usize(
                "DYN_BACKGROUND_OFFLOAD_MAX_CHUNKS",
                config.offload_max_chunks,
            )
            .max(1),
            max_inflight,
            max_serving_decode_blocks: env_u64(
                "DYN_BACKGROUND_OFFLOAD_MAX_ACTIVE_DECODE_BLOCKS",
                16,
            ),
            max_serving_prefill_tokens: env_u64(
                "DYN_BACKGROUND_OFFLOAD_MAX_PENDING_PREFILL_TOKENS",
                0,
            ),
            bytes_per_sec: env_u64("DYN_BACKGROUND_OFFLOAD_BYTES_PER_SEC", 64 * 1024 * 1024),
            chunk_bytes: env_u64("DYN_BACKGROUND_OFFLOAD_CHUNK_BYTES", 1024 * 1024).max(1),
            cooldown: Duration::from_secs(env_u64(
                "DYN_BACKGROUND_OFFLOAD_COOLDOWN_SECS",
                config.offload_cooldown_secs,
            )),
            endpoint: env::var("DYN_LMCACHE_CONTROLLER_ENDPOINT").ok(),
            instance_template: env::var("DYN_BACKGROUND_OFFLOAD_INSTANCE_TEMPLATE").ok(),
            instance_map: parse_instance_map(
                &env::var("DYN_BACKGROUND_OFFLOAD_INSTANCE_MAP").unwrap_or_default(),
            ),
            timeout: Duration::from_millis(env_u64("DYN_BACKGROUND_OFFLOAD_TIMEOUT_MS", 5_000)),
        }
    }

    fn max_bytes_per_operation(&self) -> u64 {
        let estimated = (self.max_chunks as u64).saturating_mul(self.chunk_bytes);
        if self.bytes_per_sec == 0 {
            0
        } else {
            estimated.min(self.bytes_per_sec.max(self.chunk_bytes))
        }
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<bool>().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env_u64(name, default as u64) as usize
}

fn parse_instance_map(value: &str) -> HashMap<u64, String> {
    value
        .split(',')
        .filter_map(|entry| {
            let (worker, instance) = entry.split_once(['=', ':'])?;
            let worker = worker.trim().parse::<u64>().ok()?;
            let instance = instance.trim();
            (!instance.is_empty()).then(|| (worker, instance.to_owned()))
        })
        .collect()
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct PrefixKey {
    hashes: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefixOffloadState {
    /// The prefix has a CPU source and is not fully visible in CXL.
    CpuOnly,
    /// A copy operation has been dispatched and has not completed locally.
    Copying,
    /// The prefix is fully visible in CXL according to the latest index view.
    CpuAndCxl,
}

/// Count the blocks that are continuously resident from the beginning of a
/// matched request prefix. Aggregate tier scores count blocks independently
/// and can overestimate a prefix when a worker misses an early block, so the
/// selector uses the bounded per-block metadata instead.
fn continuous_residency_blocks(
    metadata: &[BlockResidencyMetadata],
    prefix_blocks: usize,
    predicate: impl Fn(&BlockResidencyMetadata) -> bool,
) -> usize {
    (0..prefix_blocks)
        .take_while(|block_index| {
            metadata
                .iter()
                .any(|entry| entry.block_index == *block_index as u32 && predicate(entry))
        })
        .count()
}

fn cpu_coverage(overlaps: &OverlapScores, prefix_blocks: usize) -> HashMap<WorkerWithDpRank, f64> {
    if prefix_blocks == 0 {
        return HashMap::new();
    }

    overlaps
        .block_residency
        .iter()
        .filter_map(|(worker, metadata)| {
            let continuous_blocks =
                continuous_residency_blocks(metadata, prefix_blocks, |entry| entry.cpu);
            (continuous_blocks > 0)
                .then(|| (*worker, continuous_blocks as f64 / prefix_blocks as f64))
        })
        .collect()
}

fn cxl_prefix_ready(overlaps: &OverlapScores, prefix_blocks: usize) -> bool {
    prefix_blocks > 0
        && overlaps.block_residency.values().any(|metadata| {
            continuous_residency_blocks(metadata, prefix_blocks, |entry| entry.cxl) == prefix_blocks
        })
}

fn best_cpu_worker(coverage: &HashMap<WorkerWithDpRank, f64>) -> Option<(WorkerWithDpRank, f64)> {
    coverage
        .iter()
        .max_by(|(worker_a, coverage_a), (worker_b, coverage_b)| {
            coverage_a
                .total_cmp(coverage_b)
                // Prefer the lower worker identity when coverage is equal so
                // source selection does not depend on HashMap iteration order.
                .then_with(|| worker_b.cmp(worker_a))
        })
        .map(|(worker, coverage)| (*worker, *coverage))
}

#[derive(Debug, Clone, Copy)]
struct CandidateScore {
    source_worker: WorkerWithDpRank,
    skew: f64,
    priority: f64,
}

fn score_candidate(
    stat: &PrefixStat,
    hot_threshold: u32,
    skew_threshold: f64,
) -> Option<CandidateScore> {
    if stat.recent_requests < hot_threshold || stat.state != PrefixOffloadState::CpuOnly {
        return None;
    }

    let (source_worker, best_coverage) = best_cpu_worker(&stat.cpu_coverage)?;
    let mut coverages: Vec<f64> = stat.cpu_coverage.values().copied().collect();
    coverages.sort_by(|a, b| b.total_cmp(a));
    let second_best_coverage = coverages.get(1).copied().unwrap_or(0.0);
    let skew = (best_coverage - second_best_coverage).max(0.0);
    if skew < skew_threshold {
        return None;
    }

    Some(CandidateScore {
        source_worker,
        skew,
        priority: stat.recent_requests as f64 * skew,
    })
}

/// A small cross-language fingerprint used only for benchmark observability.
///
/// The offload planner already owns the exact token prefix that it sends to
/// LMCache, while the worker-side completion log owns the actual tier hit
/// counts.  Logging a deterministic token fingerprint lets the benchmark join
/// those two records without changing the wire protocol or the cache key.
fn prefix_fingerprint(tokens: &[u32]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut fingerprint = OFFSET_BASIS;
    for token in tokens {
        for byte in token.to_le_bytes() {
            fingerprint ^= u64::from(byte);
            fingerprint = fingerprint.wrapping_mul(PRIME);
        }
    }
    fingerprint
}

fn prefix_observability_id(key: &PrefixKey, tokens: &[u32]) -> String {
    format!("{}:{:016x}", key.hashes.len(), prefix_fingerprint(tokens))
}

#[derive(Debug, Clone)]
struct PrefixStat {
    tokens: Arc<[u32]>,
    cpu_coverage: HashMap<WorkerWithDpRank, f64>,
    state: PrefixOffloadState,
    recent_requests: u32,
    window_start: Instant,
    last_seen: Instant,
    last_offload: Option<Instant>,
}

#[derive(Debug, Default)]
struct PlannerState {
    prefixes: HashMap<PrefixKey, PrefixStat>,
    serving_decode_blocks: u64,
    serving_prefill_tokens: u64,
}

#[derive(Debug)]
struct ByteRateLimiter {
    bytes_per_sec: u64,
    capacity: u64,
    tokens: f64,
    last_refill: Instant,
}

impl ByteRateLimiter {
    fn new(bytes_per_sec: u64, burst_bytes: u64) -> Self {
        let capacity = if bytes_per_sec == 0 {
            0
        } else {
            bytes_per_sec.max(burst_bytes)
        };
        Self {
            bytes_per_sec,
            capacity,
            tokens: capacity as f64,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self, now: Instant) {
        if self.bytes_per_sec == 0 {
            self.last_refill = now;
            return;
        }
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        if elapsed > 0.0 {
            self.tokens =
                (self.tokens + elapsed * self.bytes_per_sec as f64).min(self.capacity as f64);
            self.last_refill = now;
        }
    }

    fn try_reserve(&mut self, bytes: u64) -> bool {
        if self.bytes_per_sec == 0 || bytes == 0 {
            return true;
        }
        self.refill(Instant::now());
        let bytes = bytes as f64;
        if self.tokens + f64::EPSILON < bytes {
            return false;
        }
        self.tokens -= bytes;
        true
    }

    fn refund(&mut self, reserved: u64, used: u64) {
        if self.bytes_per_sec == 0 || used >= reserved {
            return;
        }
        self.refill(Instant::now());
        self.tokens = (self.tokens + (reserved - used) as f64).min(self.capacity as f64);
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum OffloadRequest {
    OffloadMsg {
        event_id: String,
        instance_id: String,
        tokens: Vec<u32>,
        source: String,
        target: String,
        copy: bool,
        max_chunks: usize,
        max_bytes: u64,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum OffloadResponse {
    OffloadRetMsg {
        event_id: String,
        instance_id: String,
        success: bool,
        rank_results: Vec<RankResult>,
    },
    ErrorMsg {
        error: String,
    },
}

#[derive(Debug, Deserialize)]
struct RankResult {
    worker_id: i64,
    success: bool,
    committed_chunks: u64,
    already_present_chunks: u64,
    failed_chunks: u64,
    bytes_written: u64,
    error: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct OffloadRpcResult {
    success: bool,
    bytes_written: u64,
}

struct LmcacheOffloadClient {
    endpoint: String,
    timeout: Duration,
    socket: AsyncMutex<Option<ReqSocket>>,
}

impl std::fmt::Debug for LmcacheOffloadClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LmcacheOffloadClient")
            .field("endpoint", &self.endpoint)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

impl LmcacheOffloadClient {
    fn new(endpoint: String, timeout: Duration) -> Self {
        Self {
            endpoint,
            timeout,
            socket: AsyncMutex::new(None),
        }
    }

    async fn send(&self, request: OffloadRequest) -> Result<OffloadRpcResult> {
        let payload = to_vec_named(&request).context("encode LMCache OffloadMsg")?;
        let mut socket_guard = self.socket.lock().await;
        if socket_guard.is_none() {
            let mut socket = ReqSocket::new();
            socket
                .connect(&self.endpoint)
                .await
                .with_context(|| format!("connect LMCache controller {}", self.endpoint))?;
            *socket_guard = Some(socket);
        }
        let socket = socket_guard
            .as_mut()
            .ok_or_else(|| anyhow!("LMCache REQ socket was not initialized"))?;
        let request_result = tokio::time::timeout(self.timeout, async {
            socket
                .send(ZmqMessage::from(Bytes::from(payload)))
                .await
                .context("send LMCache OffloadMsg")?;
            let response = socket
                .recv()
                .await
                .context("receive LMCache OffloadRetMsg")?;
            let frame = response
                .into_vec()
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("LMCache controller returned an empty response"))?;
            let decoded: OffloadResponse =
                from_slice(frame.as_ref()).context("decode LMCache OffloadRetMsg")?;
            match decoded {
                OffloadResponse::OffloadRetMsg {
                    event_id,
                    instance_id,
                    success,
                    rank_results,
                } => {
                    let successful_ranks = rank_results.iter().filter(|rank| rank.success).count();
                    let committed_chunks: u64 = rank_results
                        .iter()
                        .map(|rank| rank.committed_chunks + rank.already_present_chunks)
                        .sum();
                    let failed_chunks: u64 =
                        rank_results.iter().map(|rank| rank.failed_chunks).sum();
                    let bytes_written: u64 =
                        rank_results.iter().map(|rank| rank.bytes_written).sum();
                    let rank_errors = rank_results
                        .iter()
                        .filter_map(|rank| rank.error.as_deref())
                        .collect::<Vec<_>>();
                    let rank_ids = rank_results
                        .iter()
                        .map(|rank| rank.worker_id)
                        .collect::<Vec<_>>();
                    tracing::info!(
                        %event_id,
                        %instance_id,
                        success,
                        successful_ranks,
                        total_ranks = rank_results.len(),
                        committed_chunks,
                        failed_chunks,
                        bytes_written,
                        ?rank_ids,
                        ?rank_errors,
                        "Background CXL offload response"
                    );
                    Ok(OffloadRpcResult {
                        success,
                        bytes_written,
                    })
                }
                OffloadResponse::ErrorMsg { error } => Err(anyhow!(error)),
            }
        })
        .await;
        match request_result {
            Ok(Ok(success)) => Ok(success),
            Ok(Err(error)) => {
                // Send/receive/decode failures also leave a REQ socket in an
                // unknown state.  Reconnect on the next bounded attempt.
                *socket_guard = None;
                Err(error)
            }
            Err(timeout_error) => {
                // A timed-out REQ socket cannot safely accept a new request
                // until its outstanding receive is consumed. Drop it and
                // reconnect on the next planner attempt.
                *socket_guard = None;
                Err(timeout_error.into())
            }
        }
    }
}

pub(crate) struct BackgroundOffloadPlanner {
    config: PlannerConfig,
    state: Mutex<PlannerState>,
    client: Option<Arc<LmcacheOffloadClient>>,
    inflight: Arc<Semaphore>,
    byte_limiter: Mutex<ByteRateLimiter>,
}

impl std::fmt::Debug for BackgroundOffloadPlanner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BackgroundOffloadPlanner")
            .field("config", &self.config)
            .field("client_configured", &self.client.is_some())
            .finish()
    }
}

impl BackgroundOffloadPlanner {
    pub(crate) fn from_config(router_config: KvRouterConfig) -> Option<Arc<Self>> {
        let config = PlannerConfig::from_config(router_config);
        if !config.enabled {
            return None;
        }
        if !config.dry_run
            && (config.endpoint.is_none()
                || (config.instance_template.is_none() && config.instance_map.is_empty()))
        {
            tracing::error!(
                "Background offload requires DYN_LMCACHE_CONTROLLER_ENDPOINT and an instance map/template"
            );
            return None;
        }
        let client = config
            .endpoint
            .clone()
            .map(|endpoint| Arc::new(LmcacheOffloadClient::new(endpoint, config.timeout)));
        let planner = Arc::new(Self {
            inflight: Arc::new(Semaphore::new(config.max_inflight)),
            byte_limiter: Mutex::new(ByteRateLimiter::new(
                config.bytes_per_sec,
                config.chunk_bytes,
            )),
            config,
            state: Mutex::new(PlannerState::default()),
            client,
        });
        tracing::info!(
            interval_ms = planner.config.interval.as_millis(),
            window_ms = planner.config.window.as_millis(),
            hot_threshold = planner.config.hot_threshold,
            skew_threshold = planner.config.skew_threshold,
            max_offloads_per_round = planner.config.max_offloads_per_round,
            max_inflight = planner.config.max_inflight,
            max_active_decode_blocks = planner.config.max_serving_decode_blocks,
            max_pending_prefill_tokens = planner.config.max_serving_prefill_tokens,
            bytes_per_sec = planner.config.bytes_per_sec,
            chunk_bytes = planner.config.chunk_bytes,
            dry_run = planner.config.dry_run,
            "Background CPU-to-CXL offload planner enabled"
        );
        Some(planner)
    }

    pub(crate) fn start(self: &Arc<Self>, cancellation: CancellationToken) {
        let planner = Arc::clone(self);
        tokio::spawn(async move { planner.run(cancellation).await });
    }

    pub(crate) fn observe(
        &self,
        tokens: &[u32],
        block_size: u32,
        overlaps: &OverlapScores,
        decode_blocks: &HashMap<WorkerWithDpRank, usize>,
        prefill_tokens: &HashMap<WorkerWithDpRank, usize>,
    ) {
        // Keep the serving-pressure admission guard as a resource-safety
        // signal. It does not participate in which prefix wins the CPU
        // locality ranking.
        let serving_decode_blocks = decode_blocks
            .values()
            .fold(0u64, |total, value| total.saturating_add(*value as u64));
        let serving_prefill_tokens = prefill_tokens
            .values()
            .fold(0u64, |total, value| total.saturating_add(*value as u64));
        let Ok(mut state) = self.state.lock() else {
            tracing::warn!("Background offload planner state mutex poisoned");
            return;
        };
        state.serving_decode_blocks = serving_decode_blocks;
        state.serving_prefill_tokens = serving_prefill_tokens;
        drop(state);
        if tokens.is_empty() || block_size == 0 {
            return;
        }
        let hashes = compute_block_hash_for_seq(tokens, block_size, None);
        let chunk_count = hashes.len().min(self.config.max_chunks);
        if chunk_count == 0 {
            return;
        }
        let prefix_tokens = tokens[..chunk_count * block_size as usize].to_vec();
        let key = PrefixKey {
            hashes: hashes[..chunk_count].iter().map(|hash| hash.0).collect(),
        };
        let current_cpu_coverage = cpu_coverage(overlaps, chunk_count);
        let cxl_ready = cxl_prefix_ready(overlaps, chunk_count);
        let now = Instant::now();
        let Ok(mut state) = self.state.lock() else {
            tracing::warn!("Background offload planner state mutex poisoned");
            return;
        };
        let stat = state.prefixes.entry(key).or_insert_with(|| PrefixStat {
            tokens: Arc::from(prefix_tokens.clone()),
            cpu_coverage: current_cpu_coverage.clone(),
            state: if cxl_ready {
                PrefixOffloadState::CpuAndCxl
            } else {
                PrefixOffloadState::CpuOnly
            },
            recent_requests: 0,
            window_start: now,
            last_seen: now,
            last_offload: None,
        });
        if now.duration_since(stat.window_start) > self.config.window {
            stat.recent_requests = 0;
            stat.window_start = now;
        }
        stat.recent_requests = stat.recent_requests.saturating_add(1);
        stat.tokens = Arc::from(prefix_tokens);
        // Keep the latest CPU coverage rather than averaging historical
        // observations: CPU residency changes with eviction and new writes.
        stat.cpu_coverage = current_cpu_coverage;
        // Do not overwrite Copying until event-backed CXL visibility is
        // confirmed by a later route observation.
        if stat.state == PrefixOffloadState::Copying {
            // Keep a successful operation suppressed until the normal CXL
            // event/index path confirms that the complete prefix is visible.
            if cxl_ready {
                stat.state = PrefixOffloadState::CpuAndCxl;
            }
        } else {
            stat.state = if cxl_ready {
                PrefixOffloadState::CpuAndCxl
            } else {
                PrefixOffloadState::CpuOnly
            };
        }
        stat.last_seen = now;
    }

    async fn run(self: Arc<Self>, cancellation: CancellationToken) {
        let mut interval = tokio::time::interval(self.config.interval);
        loop {
            tokio::select! {
                _ = cancellation.cancelled() => return,
                _ = interval.tick() => self.plan_once().await,
            }
        }
    }

    async fn plan_once(self: &Arc<Self>) {
        let now = Instant::now();
        let mut candidates: Vec<(PrefixKey, Arc<[u32]>, WorkerWithDpRank, u32, f64, f64)> =
            Vec::new();
        {
            let Ok(mut state) = self.state.lock() else {
                tracing::warn!("Background offload planner state mutex poisoned");
                return;
            };
            let serving_admissible = state.serving_decode_blocks
                <= self.config.max_serving_decode_blocks
                && state.serving_prefill_tokens <= self.config.max_serving_prefill_tokens;
            if !serving_admissible {
                tracing::debug!(
                    serving_decode_blocks = state.serving_decode_blocks,
                    serving_prefill_tokens = state.serving_prefill_tokens,
                    max_active_decode_blocks = self.config.max_serving_decode_blocks,
                    max_pending_prefill_tokens = self.config.max_serving_prefill_tokens,
                    "Background CXL offload deferred by serving-aware admission"
                );
                return;
            }
            state
                .prefixes
                .retain(|_, stat| now.duration_since(stat.last_seen) <= self.config.window * 3);
            for (key, stat) in state.prefixes.iter_mut() {
                if now.duration_since(stat.window_start) > self.config.window {
                    stat.recent_requests = 0;
                    stat.window_start = now;
                    continue;
                }
                let cooldown_done = stat
                    .last_offload
                    .map(|last| now.duration_since(last) >= self.config.cooldown)
                    .unwrap_or(true);
                if !cooldown_done {
                    continue;
                }
                let Some(score) =
                    score_candidate(stat, self.config.hot_threshold, self.config.skew_threshold)
                else {
                    continue;
                };
                candidates.push((
                    key.clone(),
                    stat.tokens.clone(),
                    score.source_worker,
                    stat.recent_requests,
                    score.skew,
                    score.priority,
                ));
            }
        }
        candidates.sort_by(|a, b| {
            b.5.total_cmp(&a.5)
                .then_with(|| b.4.total_cmp(&a.4))
                .then_with(|| b.3.cmp(&a.3))
                .then_with(|| b.1.len().cmp(&a.1.len()))
                .then_with(|| a.0.hashes.cmp(&b.0.hashes))
        });
        candidates.truncate(self.config.max_offloads_per_round);
        for (key, tokens, source_worker, _, _, _) in candidates {
            // Reserve both the bounded execution slot and the byte budget
            // before changing candidate state. A failed admission therefore
            // remains eligible on the next planner tick.
            let Ok(permit) = Arc::clone(&self.inflight).try_acquire_owned() else {
                break;
            };
            let reserved_bytes = self.config.max_bytes_per_operation();
            let budget_reserved = self
                .byte_limiter
                .lock()
                .map(|mut limiter| limiter.try_reserve(reserved_bytes))
                .unwrap_or(false);
            if !budget_reserved {
                drop(permit);
                tracing::debug!(
                    bytes = reserved_bytes,
                    bytes_per_sec = self.config.bytes_per_sec,
                    "Background CXL offload deferred by byte-rate admission"
                );
                continue;
            }
            // Mark only the candidates that survived priority ranking. A
            // failed admission remains eligible on the next planner tick.
            let marked = self
                .state
                .lock()
                .ok()
                .and_then(|mut state| {
                    let stat = state.prefixes.get_mut(&key)?;
                    if stat.state != PrefixOffloadState::CpuOnly {
                        return Some(false);
                    }
                    stat.state = PrefixOffloadState::Copying;
                    stat.last_offload = Some(now);
                    Some(true)
                })
                .unwrap_or(false);
            if !marked {
                if let Ok(mut limiter) = self.byte_limiter.lock() {
                    limiter.refund(reserved_bytes, 0);
                }
                drop(permit);
                continue;
            }
            let planner = Arc::clone(self);
            tokio::spawn(async move {
                let _permit = permit;
                planner
                    .execute(key, tokens, source_worker, reserved_bytes)
                    .await;
            });
        }
    }

    fn clear_inflight(&self, key: &PrefixKey) {
        if let Ok(mut state) = self.state.lock()
            && let Some(stat) = state.prefixes.get_mut(key)
        {
            if stat.state == PrefixOffloadState::Copying {
                stat.state = PrefixOffloadState::CpuOnly;
            }
        }
    }

    async fn execute(
        &self,
        key: PrefixKey,
        tokens: Arc<[u32]>,
        owner: WorkerWithDpRank,
        reserved_bytes: u64,
    ) {
        let prefix_id = prefix_observability_id(&key, tokens.as_ref());
        let event_id = format!("background-offload-{}", uuid::Uuid::new_v4());
        tracing::info!(
            %event_id,
            %prefix_id,
            worker_id = owner.worker_id,
            dp_rank = owner.dp_rank,
            chunks = key.hashes.len(),
            tokens = tokens.len(),
            "Background CXL offload dispatch"
        );
        let result: Result<OffloadRpcResult> = if self.config.dry_run {
            tracing::info!(
                worker_id = owner.worker_id,
                dp_rank = owner.dp_rank,
                chunks = key.hashes.len(),
                tokens = tokens.len(),
                %prefix_id,
                "Background CXL offload dry-run candidate"
            );
            Ok(OffloadRpcResult {
                success: false,
                bytes_written: 0,
            })
        } else {
            match (&self.client, self.instance_id(owner)) {
                (Some(client), Some(instance_id)) => {
                    client
                        .send(OffloadRequest::OffloadMsg {
                            event_id,
                            instance_id,
                            tokens: tokens.to_vec(),
                            source: "LocalCPUBackend".to_string(),
                            target: "CxlBackend".to_string(),
                            copy: true,
                            max_chunks: key.hashes.len().min(self.config.max_chunks),
                            max_bytes: reserved_bytes,
                        })
                        .await
                }
                (None, _) => Err(anyhow!("LMCache offload client is not configured")),
                (_, None) => Err(anyhow!(
                    "no LMCache instance mapping for worker {}",
                    owner.worker_id
                )),
            }
        };
        if let Ok(rpc_result) = result {
            if let Ok(mut limiter) = self.byte_limiter.lock() {
                limiter.refund(reserved_bytes, rpc_result.bytes_written);
            }
            if !rpc_result.success {
                tracing::warn!(
                    worker_id = owner.worker_id,
                    dp_rank = owner.dp_rank,
                    bytes_written = rpc_result.bytes_written,
                    "Background CXL offload completed without CXL_READY success"
                );
            }
            // The RPC result only describes rank-local writes. CXL
            // routability is still determined by the normal event/index path;
            // a later observation will move the prefix to CpuAndCxl when the
            // complete prefix is visible. Release the execution state here so
            // a lost event cannot strand the prefix forever; last_offload keeps
            // retries behind the cooldown.
            self.clear_inflight(&key);
        } else {
            if let Ok(mut limiter) = self.byte_limiter.lock() {
                limiter.refund(reserved_bytes, 0);
            }
            self.clear_inflight(&key);
        }
    }

    fn instance_id(&self, worker: WorkerWithDpRank) -> Option<String> {
        self.config
            .instance_map
            .get(&worker.worker_id)
            .cloned()
            .or_else(|| {
                self.config.instance_template.as_ref().map(|template| {
                    template
                        .replace("{worker_id}", &worker.worker_id.to_string())
                        .replace("{dp_rank}", &worker.dp_rank.to_string())
                })
            })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use tokio::sync::Semaphore;

    use super::{
        BackgroundOffloadPlanner, ByteRateLimiter, PlannerConfig, PlannerState, PrefixKey,
        PrefixOffloadState, PrefixStat, continuous_residency_blocks, parse_instance_map,
        score_candidate,
    };
    use crate::kv_router::{
        indexer::{BlockResidencyMetadata, OverlapScores},
        protocols::{LocalBlockHash, WorkerWithDpRank},
    };

    fn test_config() -> PlannerConfig {
        PlannerConfig {
            enabled: true,
            dry_run: true,
            interval: Duration::from_secs(5),
            window: Duration::from_secs(10),
            hot_threshold: 3,
            skew_threshold: 0.5,
            max_offloads_per_round: 2,
            max_chunks: 16,
            max_inflight: 2,
            max_serving_decode_blocks: 8,
            max_serving_prefill_tokens: 0,
            bytes_per_sec: 1024,
            chunk_bytes: 1024,
            cooldown: Duration::from_secs(30),
            endpoint: None,
            instance_template: None,
            instance_map: HashMap::new(),
            timeout: Duration::from_secs(5),
        }
    }

    fn test_planner() -> BackgroundOffloadPlanner {
        let config = test_config();
        let max_inflight = config.max_inflight;
        BackgroundOffloadPlanner {
            config,
            state: std::sync::Mutex::new(PlannerState::default()),
            client: None,
            inflight: Arc::new(Semaphore::new(max_inflight)),
            byte_limiter: std::sync::Mutex::new(ByteRateLimiter::new(1024, 1024)),
        }
    }

    fn residency(block_index: usize, gpu: bool, cpu: bool, cxl: bool) -> BlockResidencyMetadata {
        BlockResidencyMetadata {
            block_index: block_index as u32,
            block_hash: LocalBlockHash(block_index as u64),
            gpu,
            cpu,
            cxl,
            prefetch_inflight: false,
        }
    }

    fn add_residency(
        overlaps: &mut OverlapScores,
        worker: WorkerWithDpRank,
        prefix_blocks: usize,
        cpu_blocks: usize,
        cxl_blocks: usize,
        gpu: bool,
    ) {
        overlaps.block_residency.insert(
            worker,
            (0..prefix_blocks)
                .map(|block_index| {
                    residency(
                        block_index,
                        gpu,
                        block_index < cpu_blocks,
                        block_index < cxl_blocks,
                    )
                })
                .collect(),
        );
    }

    fn prefix_stat(
        request_count: u32,
        cpu_coverage: HashMap<WorkerWithDpRank, f64>,
        state: PrefixOffloadState,
    ) -> PrefixStat {
        let now = Instant::now();
        PrefixStat {
            tokens: Arc::from(vec![1, 2, 3, 4]),
            cpu_coverage,
            state,
            recent_requests: request_count,
            window_start: now,
            last_seen: now,
            last_offload: None,
        }
    }

    #[test]
    fn parses_static_worker_instance_mapping() {
        let map = parse_instance_map("10=lmcache-a, 20:lmcache-b,invalid");
        assert_eq!(map.get(&10).map(String::as_str), Some("lmcache-a"));
        assert_eq!(map.get(&20).map(String::as_str), Some("lmcache-b"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn byte_rate_limiter_requires_budget_and_refunds_unused_bytes() {
        let mut limiter = ByteRateLimiter::new(100, 100);
        assert!(limiter.try_reserve(100));
        assert!(!limiter.try_reserve(1));
        limiter.refund(100, 40);
        assert!(limiter.try_reserve(60));
        assert!(!limiter.try_reserve(1));
    }

    #[test]
    fn observes_aggregate_serving_pressure() {
        let planner = test_planner();
        let owner = WorkerWithDpRank::from_worker_id(21);
        let overlaps = OverlapScores::new();
        let mut decode_blocks = HashMap::new();
        decode_blocks.insert(owner, 7);
        let mut prefill_tokens = HashMap::new();
        prefill_tokens.insert(owner, 33);

        planner.observe(&[1, 2, 3, 4], 4, &overlaps, &decode_blocks, &prefill_tokens);

        let state = planner.state.lock().expect("planner state lock");
        assert_eq!(state.serving_decode_blocks, 7);
        assert_eq!(state.serving_prefill_tokens, 33);
    }

    #[test]
    fn observes_only_chunk_aligned_prefix_tokens() {
        let planner = test_planner();
        let owner = WorkerWithDpRank::from_worker_id(7);
        let mut overlaps = OverlapScores::new();
        add_residency(&mut overlaps, owner, 2, 2, 0, false);

        planner.observe(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9],
            4,
            &overlaps,
            &HashMap::new(),
            &HashMap::new(),
        );

        let state = planner.state.lock().expect("planner state lock");
        let stat: &PrefixStat = state.prefixes.values().next().expect("prefix stat");
        assert_eq!(stat.tokens.as_ref(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(stat.recent_requests, 1);
        assert_eq!(stat.cpu_coverage.get(&owner), Some(&1.0));
        assert_eq!(stat.state, PrefixOffloadState::CpuOnly);
    }

    #[test]
    fn observes_gpu_and_cpu_prefix_when_cxl_is_absent() {
        let planner = test_planner();
        let owner = WorkerWithDpRank::from_worker_id(11);
        let mut overlaps = OverlapScores::new();
        add_residency(&mut overlaps, owner, 2, 2, 0, true);

        planner.observe(
            &[1, 2, 3, 4, 5, 6, 7, 8],
            4,
            &overlaps,
            &HashMap::new(),
            &HashMap::new(),
        );

        let state = planner.state.lock().expect("planner state lock");
        let stat = state.prefixes.values().next().expect("prefix stat");
        assert_eq!(stat.recent_requests, 1);
        assert_eq!(stat.cpu_coverage.get(&owner), Some(&1.0));
        assert_eq!(stat.state, PrefixOffloadState::CpuOnly);
    }

    #[test]
    fn continuous_cpu_coverage_stops_at_the_first_missing_block() {
        let metadata = vec![
            residency(0, false, true, false),
            residency(1, false, false, false),
            residency(2, false, true, false),
        ];
        assert_eq!(
            continuous_residency_blocks(&metadata, 3, |entry| entry.cpu),
            1
        );
    }

    #[test]
    fn records_cxl_residency_even_for_gpu_and_cpu_prefixes() {
        let planner = test_planner();
        let owner = WorkerWithDpRank::from_worker_id(12);
        let mut overlaps = OverlapScores::new();
        add_residency(&mut overlaps, owner, 2, 2, 2, true);

        planner.observe(
            &[1, 2, 3, 4, 5, 6, 7, 8],
            4,
            &overlaps,
            &HashMap::new(),
            &HashMap::new(),
        );

        let state = planner.state.lock().expect("planner state lock");
        let stat = state.prefixes.values().next().expect("prefix stat");
        assert_eq!(stat.state, PrefixOffloadState::CpuAndCxl);
    }

    #[test]
    fn candidate_score_requires_frequency_and_skew_and_uses_product_priority() {
        let best = WorkerWithDpRank::from_worker_id(1);
        let second = WorkerWithDpRank::from_worker_id(2);
        let stat = prefix_stat(
            10,
            HashMap::from([(best, 1.0), (second, 0.2)]),
            PrefixOffloadState::CpuOnly,
        );
        let score = score_candidate(&stat, 3, 0.5).expect("eligible candidate");
        assert_eq!(score.source_worker, best);
        assert!((score.skew - 0.8).abs() < f64::EPSILON);
        assert!((score.priority - 8.0).abs() < f64::EPSILON);

        assert!(
            score_candidate(
                &prefix_stat(2, stat.cpu_coverage.clone(), stat.state),
                3,
                0.5
            )
            .is_none()
        );
        assert!(
            score_candidate(
                &prefix_stat(
                    10,
                    HashMap::from([(best, 1.0), (second, 0.8)]),
                    PrefixOffloadState::CpuOnly,
                ),
                3,
                0.5,
            )
            .is_none()
        );
        assert!(
            score_candidate(
                &prefix_stat(
                    10,
                    HashMap::from([(best, 1.0), (second, 0.2)]),
                    PrefixOffloadState::CpuAndCxl,
                ),
                3,
                0.5,
            )
            .is_none()
        );
    }

    #[tokio::test]
    async fn serving_pressure_defers_candidate_before_copy() {
        let planner = Arc::new(test_planner());
        let worker = WorkerWithDpRank::from_worker_id(3);
        let key = PrefixKey { hashes: vec![42] };
        let stat = prefix_stat(
            3,
            HashMap::from([(worker, 1.0)]),
            PrefixOffloadState::CpuOnly,
        );
        {
            let mut state = planner.state.lock().expect("planner state lock");
            state.serving_decode_blocks = 9;
            state.prefixes.insert(key.clone(), stat);
        }

        planner.plan_once().await;

        let state = planner.state.lock().expect("planner state lock");
        assert_eq!(state.prefixes[&key].state, PrefixOffloadState::CpuOnly);
    }
}
