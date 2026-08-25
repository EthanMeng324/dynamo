// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Periodic, router-owned CPU-to-CXL offload planning.
//!
//! Request handling only records observations.  A bounded background task
//! wakes every few seconds, ranks hot prefixes, and sends a MessagePack
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
    KvRouterConfig, compute_block_hash_for_seq, indexer::OverlapScores, protocols::WorkerWithDpRank,
};

#[derive(Debug, Clone)]
struct PlannerConfig {
    enabled: bool,
    dry_run: bool,
    interval: Duration,
    window: Duration,
    hot_threshold: u32,
    owner_load_threshold: u64,
    top_k: usize,
    max_chunks: usize,
    /// Kept for wire/config compatibility; admission is deliberately single
    /// concurrent so a background copy cannot create a burst of CXL traffic.
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
            owner_load_threshold: env_u64(
                "DYN_BACKGROUND_OFFLOAD_OWNER_LOAD_THRESHOLD",
                config.offload_owner_load_threshold,
            ),
            top_k: env_usize("DYN_BACKGROUND_OFFLOAD_TOP_K", config.offload_top_k).max(1),
            max_chunks: env_usize(
                "DYN_BACKGROUND_OFFLOAD_MAX_CHUNKS",
                config.offload_max_chunks,
            )
            .max(1),
            // Automatic offload is intentionally serialized.  Keep reading
            // the old setting so existing configurations remain parseable,
            // but do not let it reintroduce concurrent CXL copies.
            max_inflight: 1,
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

#[derive(Debug, Clone)]
struct PrefixStat {
    tokens: Arc<[u32]>,
    owner_worker: WorkerWithDpRank,
    owner_load: u64,
    cxl_ready: bool,
    recent_requests: u32,
    window_start: Instant,
    last_seen: Instant,
    last_offload: Option<Instant>,
    inflight: bool,
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
            // Single background copy at a time is part of the admission
            // contract, independent of the legacy max-inflight knob.
            inflight: Arc::new(Semaphore::new(1)),
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
            owner_load_threshold = planner.config.owner_load_threshold,
            max_active_decode_blocks = planner.config.max_serving_decode_blocks,
            max_pending_prefill_tokens = planner.config.max_serving_prefill_tokens,
            bytes_per_sec = planner.config.bytes_per_sec,
            chunk_bytes = planner.config.chunk_bytes,
            max_inflight = planner.config.max_inflight,
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
        _best_worker: WorkerWithDpRank,
        overlaps: &OverlapScores,
        decode_blocks: &HashMap<WorkerWithDpRank, usize>,
        prefill_tokens: &HashMap<WorkerWithDpRank, usize>,
    ) {
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
        // A source prefix is usable for offload when its CPU copy is visible
        // either as CPU-only or alongside a GPU copy.  The latter is common
        // before GPU eviction and must not be excluded merely because the
        // scheduler gives GPU residency its own routing bucket.
        let mut source_scores = HashMap::new();
        for (worker, score) in overlaps
            .cpu_scores
            .iter()
            .chain(overlaps.gpu_and_cpu_scores.iter())
        {
            let entry = source_scores.entry(*worker).or_insert(0u32);
            *entry = entry.saturating_add(*score);
        }
        let Some(source_worker) = source_scores
            .iter()
            .max_by_key(|(_, score)| *score)
            .filter(|(_, score)| **score > 0)
            .map(|(worker, _)| *worker)
        else {
            // A GPU-only/no-cache route has no LocalCPU source and should not
            // trigger a futile CPU lookup/offload attempt.
            return;
        };
        let owner_load = decode_blocks
            .get(&source_worker)
            .copied()
            .unwrap_or_default() as u64
            + prefill_tokens
                .get(&source_worker)
                .copied()
                .unwrap_or_default()
                .div_ceil(block_size as usize) as u64;
        let cxl_ready = overlaps
            .cxl_resident_scores
            .get(&source_worker)
            .copied()
            .unwrap_or(0)
            > 0
            || overlaps
                .cxl_scores
                .get(&source_worker)
                .copied()
                .unwrap_or(0)
                > 0
            || overlaps
                .cpu_and_cxl_scores
                .get(&source_worker)
                .copied()
                .unwrap_or(0)
                > 0;
        let now = Instant::now();
        let Ok(mut state) = self.state.lock() else {
            tracing::warn!("Background offload planner state mutex poisoned");
            return;
        };
        let stat = state.prefixes.entry(key).or_insert_with(|| PrefixStat {
            tokens: Arc::from(prefix_tokens.clone()),
            owner_worker: source_worker,
            owner_load,
            cxl_ready,
            recent_requests: 0,
            window_start: now,
            last_seen: now,
            last_offload: None,
            inflight: false,
        });
        if now.duration_since(stat.window_start) > self.config.window {
            stat.recent_requests = 0;
            stat.window_start = now;
        }
        stat.recent_requests = stat.recent_requests.saturating_add(1);
        stat.tokens = Arc::from(prefix_tokens);
        stat.owner_worker = source_worker;
        stat.owner_load = owner_load;
        // Residency is driven by the latest index observation.  Do not keep
        // a stale `true` after CXL eviction/removal; the next hot window may
        // legitimately admit the prefix again.
        stat.cxl_ready = cxl_ready;
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
        let mut candidates = Vec::new();
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
                if stat.recent_requests >= self.config.hot_threshold
                    && stat.owner_load >= self.config.owner_load_threshold
                    && !stat.cxl_ready
                    && !stat.inflight
                    && cooldown_done
                {
                    candidates.push((
                        key.clone(),
                        stat.tokens.clone(),
                        stat.owner_worker,
                        stat.recent_requests,
                    ));
                }
            }
        }
        candidates.sort_by_key(|(_, tokens, _, request_count)| {
            (
                std::cmp::Reverse(*request_count),
                std::cmp::Reverse(tokens.len()),
            )
        });
        candidates.truncate(self.config.top_k);
        for (key, tokens, owner, _) in candidates {
            // Reserve both the single execution slot and the byte budget
            // before changing candidate state.  A failed admission therefore
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
            // Mark only the candidates that survived top-K ranking.  Marking
            // the complete pre-truncation set would strand entries in
            // `inflight=true` forever when top-K or the semaphore drops them.
            let marked = self
                .state
                .lock()
                .ok()
                .and_then(|mut state| {
                    let stat = state.prefixes.get_mut(&key)?;
                    if stat.inflight {
                        return Some(false);
                    }
                    stat.inflight = true;
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
                planner.execute(key, tokens, owner, reserved_bytes).await;
            });
            // The semaphore is intentionally one-wide.  Avoid walking the
            // remaining top-K candidates only to mark and immediately clear
            // them while this operation is active.
            break;
        }
    }

    fn clear_inflight(&self, key: &PrefixKey) {
        if let Ok(mut state) = self.state.lock()
            && let Some(stat) = state.prefixes.get_mut(key)
        {
            stat.inflight = false;
        }
    }

    async fn execute(
        &self,
        key: PrefixKey,
        tokens: Arc<[u32]>,
        owner: WorkerWithDpRank,
        reserved_bytes: u64,
    ) {
        let result: Result<OffloadRpcResult> = if self.config.dry_run {
            tracing::info!(
                worker_id = owner.worker_id,
                dp_rank = owner.dp_rank,
                chunks = key.hashes.len(),
                tokens = tokens.len(),
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
                            event_id: format!("background-offload-{}", uuid::Uuid::new_v4()),
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
            if let Ok(mut state) = self.state.lock()
                && let Some(stat) = state.prefixes.get_mut(&key)
            {
                stat.inflight = false;
                // An RPC success only means that the rank-local writes
                // completed.  Keep routing disabled until a real CXL
                // CacheStoreEvent is observed on a subsequent request.
            }
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
    use std::time::Duration;

    use tokio::sync::Semaphore;

    use super::{
        BackgroundOffloadPlanner, ByteRateLimiter, PlannerConfig, PlannerState, PrefixStat,
        parse_instance_map,
    };
    use crate::kv_router::{indexer::OverlapScores, protocols::WorkerWithDpRank};

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
        let config = PlannerConfig {
            enabled: true,
            dry_run: true,
            interval: Duration::from_secs(5),
            window: Duration::from_secs(10),
            hot_threshold: 1,
            owner_load_threshold: 0,
            top_k: 100,
            max_chunks: 8,
            max_inflight: 1,
            max_serving_decode_blocks: 8,
            max_serving_prefill_tokens: 0,
            bytes_per_sec: 0,
            chunk_bytes: 1024,
            cooldown: Duration::from_secs(30),
            endpoint: None,
            instance_template: None,
            instance_map: HashMap::new(),
            timeout: Duration::from_secs(5),
        };
        let planner = BackgroundOffloadPlanner {
            config,
            state: std::sync::Mutex::new(PlannerState::default()),
            client: None,
            inflight: Arc::new(Semaphore::new(1)),
            byte_limiter: std::sync::Mutex::new(ByteRateLimiter::new(0, 1024)),
        };
        let owner = WorkerWithDpRank::from_worker_id(21);
        let mut overlaps = OverlapScores::new();
        overlaps.cpu_scores.insert(owner, 4);
        let mut decode_blocks = HashMap::new();
        decode_blocks.insert(owner, 7);
        let mut prefill_tokens = HashMap::new();
        prefill_tokens.insert(owner, 33);

        planner.observe(
            &[1, 2, 3, 4],
            4,
            owner,
            &overlaps,
            &decode_blocks,
            &prefill_tokens,
        );

        let state = planner.state.lock().expect("planner state lock");
        assert_eq!(state.serving_decode_blocks, 7);
        assert_eq!(state.serving_prefill_tokens, 33);
    }

    #[test]
    fn observes_only_chunk_aligned_prefix_tokens() {
        let config = PlannerConfig {
            enabled: true,
            dry_run: true,
            interval: Duration::from_secs(5),
            window: Duration::from_secs(10),
            hot_threshold: 5,
            owner_load_threshold: 10,
            top_k: 100,
            max_chunks: 8,
            max_inflight: 1,
            max_serving_decode_blocks: u64::MAX,
            max_serving_prefill_tokens: u64::MAX,
            bytes_per_sec: 0,
            chunk_bytes: 1024,
            cooldown: Duration::from_secs(30),
            endpoint: None,
            instance_template: None,
            instance_map: HashMap::new(),
            timeout: Duration::from_secs(5),
        };
        let planner = BackgroundOffloadPlanner {
            config,
            state: std::sync::Mutex::new(PlannerState::default()),
            client: None,
            inflight: Arc::new(Semaphore::new(1)),
            byte_limiter: std::sync::Mutex::new(ByteRateLimiter::new(0, 1024)),
        };
        let owner = WorkerWithDpRank::from_worker_id(7);
        let mut overlaps = OverlapScores::new();
        overlaps.cpu_scores.insert(owner, 4);
        let mut decode_blocks = HashMap::new();
        decode_blocks.insert(owner, 12);

        planner.observe(
            &[1, 2, 3, 4, 5, 6],
            4,
            owner,
            &overlaps,
            &decode_blocks,
            &HashMap::new(),
        );

        let state = planner.state.lock().expect("planner state lock");
        let stat: &PrefixStat = state.prefixes.values().next().expect("prefix stat");
        assert_eq!(stat.tokens.as_ref(), &[1, 2, 3, 4]);
        assert_eq!(stat.recent_requests, 1);
        assert_eq!(stat.owner_load, 12);
        assert!(!stat.cxl_ready);
    }

    #[test]
    fn observes_gpu_and_cpu_prefix_when_cxl_is_absent() {
        let config = PlannerConfig {
            enabled: true,
            dry_run: true,
            interval: Duration::from_secs(5),
            window: Duration::from_secs(10),
            hot_threshold: 1,
            owner_load_threshold: 0,
            top_k: 100,
            max_chunks: 8,
            max_inflight: 1,
            max_serving_decode_blocks: u64::MAX,
            max_serving_prefill_tokens: u64::MAX,
            bytes_per_sec: 0,
            chunk_bytes: 1024,
            cooldown: Duration::from_secs(30),
            endpoint: None,
            instance_template: None,
            instance_map: HashMap::new(),
            timeout: Duration::from_secs(5),
        };
        let planner = BackgroundOffloadPlanner {
            config,
            state: std::sync::Mutex::new(PlannerState::default()),
            client: None,
            inflight: Arc::new(Semaphore::new(1)),
            byte_limiter: std::sync::Mutex::new(ByteRateLimiter::new(0, 1024)),
        };
        let owner = WorkerWithDpRank::from_worker_id(11);
        let mut overlaps = OverlapScores::new();
        overlaps.gpu_and_cpu_scores.insert(owner, 4);
        let mut decode_blocks = HashMap::new();
        decode_blocks.insert(owner, 3);

        planner.observe(
            &[1, 2, 3, 4, 5, 6],
            4,
            owner,
            &overlaps,
            &decode_blocks,
            &HashMap::new(),
        );

        let state = planner.state.lock().expect("planner state lock");
        let stat = state.prefixes.values().next().expect("prefix stat");
        assert_eq!(stat.owner_worker, owner);
        assert_eq!(stat.recent_requests, 1);
        assert_eq!(stat.owner_load, 3);
        assert!(!stat.cxl_ready);
    }

    #[test]
    fn records_cxl_residency_even_for_gpu_and_cpu_prefixes() {
        let config = PlannerConfig {
            enabled: true,
            dry_run: true,
            interval: Duration::from_secs(5),
            window: Duration::from_secs(10),
            hot_threshold: 1,
            owner_load_threshold: 0,
            top_k: 100,
            max_chunks: 8,
            max_inflight: 1,
            max_serving_decode_blocks: u64::MAX,
            max_serving_prefill_tokens: u64::MAX,
            bytes_per_sec: 0,
            chunk_bytes: 1024,
            cooldown: Duration::from_secs(30),
            endpoint: None,
            instance_template: None,
            instance_map: HashMap::new(),
            timeout: Duration::from_secs(5),
        };
        let planner = BackgroundOffloadPlanner {
            config,
            state: std::sync::Mutex::new(PlannerState::default()),
            client: None,
            inflight: Arc::new(Semaphore::new(1)),
            byte_limiter: std::sync::Mutex::new(ByteRateLimiter::new(0, 1024)),
        };
        let owner = WorkerWithDpRank::from_worker_id(12);
        let mut overlaps = OverlapScores::new();
        overlaps.gpu_and_cpu_scores.insert(owner, 4);
        overlaps.cxl_resident_scores.insert(owner, 4);

        planner.observe(
            &[1, 2, 3, 4, 5, 6],
            4,
            owner,
            &overlaps,
            &HashMap::new(),
            &HashMap::new(),
        );

        let state = planner.state.lock().expect("planner state lock");
        let stat = state.prefixes.values().next().expect("prefix stat");
        assert!(stat.cxl_ready);
    }
}
