//! Router-owned global chunk association prefetch manager.
//!
//! The manager deliberately contains no KV payload and performs no local
//! storage operation. It learns associations between individual KV chunks
//! observed in consecutive requests and sends best-effort commands to the
//! selected LMCache worker. LMCache remains the data-plane executor.
//! This feature would not be used 

use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::prefetch_hint::{PrefetchFailureCallback, PrefetchHintDispatcher};
use super::protocols::{LocalBlockHash, WorkerWithDpRank, compute_block_hash_for_seq};

#[derive(Debug, Clone)]
struct GlobalPrefetchConfig {
    enabled: bool,
    chunk_size: usize,
    top_k: usize,
    min_support: f64,
    min_probability: f64,
    half_life: Duration,
    session_ttl: Duration,
    max_sessions: usize,
    max_chunk_catalog: usize,
    max_observation_chunks: usize,
    max_inflight: usize,
    current_request_enabled: bool,
    temporal_enabled: bool,
    /// Ignore chunks that have appeared in several distinct sessions when
    /// learning and querying associations. Such chunks are normally shared
    /// system/template prefixes and are poor request-specific predictors.
    filter_stable_chunks: bool,
    stable_session_threshold: usize,
    /// Estimated prefetch throughput in chunks per second. This is only used
    /// to put queue depth and copy work in the same units; it is deliberately
    /// independent of Dynamo's routing/load score.
    chunks_per_second: f64,
    ttl: Duration,
}

impl GlobalPrefetchConfig {
    fn from_env() -> Self {
        let chunk_size = env_usize("DYN_GLOBAL_PREFETCH_CHUNK_SIZE", 256).max(1);
        Self {
            enabled: env_bool("DYN_GLOBAL_PREFETCH_ENABLED", false),
            chunk_size,
            top_k: env_usize("DYN_GLOBAL_PREFETCH_TOP_K", 4).max(1),
            // Support is an exponentially decayed score, so a useful
            // functional-test threshold may be fractional.  Keep it
            // positive and bounded while leaving the production default at
            // 16 observations.
            min_support: env_f64("DYN_GLOBAL_PREFETCH_MIN_SUPPORT", 16.0).max(0.01),
            min_probability: env_f64("DYN_GLOBAL_PREFETCH_MIN_PROBABILITY", 0.65).clamp(0.01, 1.0),
            half_life: Duration::from_secs_f64(
                env_f64("DYN_GLOBAL_PREFETCH_DECAY_HALF_LIFE_SECONDS", 300.0).max(1.0),
            ),
            session_ttl: Duration::from_secs_f64(
                env_f64("DYN_GLOBAL_PREFETCH_SESSION_TTL_SECONDS", 3600.0).max(1.0),
            ),
            max_sessions: env_usize("DYN_GLOBAL_PREFETCH_MAX_SESSIONS", 10_000).max(1),
            // Keep the old environment name for deployment compatibility; it
            // now bounds chunk descriptors.
            max_chunk_catalog: env_usize("DYN_GLOBAL_PREFETCH_MAX_CATALOG", 10_000).max(1),
            max_observation_chunks: env_usize("DYN_GLOBAL_PREFETCH_MAX_OBSERVATION_CHUNKS", 64)
                .max(1),
            max_inflight: env_usize("DYN_GLOBAL_PREFETCH_MAX_INFLIGHT", 32).max(1),
            current_request_enabled: env_bool("DYN_GLOBAL_PREFETCH_CURRENT_REQUEST_ENABLED", true),
            temporal_enabled: env_bool("DYN_GLOBAL_PREFETCH_TEMPORAL_ENABLED", true),
            // Global prefetch is itself default-off. Once explicitly enabled,
            // stable-prefix filtering is the safe default; set this variable
            // to false only for an intentional legacy/permissive comparison.
            filter_stable_chunks: env_bool("DYN_GLOBAL_PREFETCH_FILTER_STABLE_CHUNKS", true),
            stable_session_threshold: env_usize("DYN_GLOBAL_PREFETCH_STABLE_SESSION_THRESHOLD", 4)
                .max(2),
            chunks_per_second: env_f64("DYN_GLOBAL_PREFETCH_CHUNKS_PER_SECOND", 64.0),
            ttl: Duration::from_millis(env_usize("DYN_GLOBAL_PREFETCH_TTL_MS", 5000).max(1) as u64),
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

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(default)
}

type ChunkKey = (String, u64);

/// Approximate per-worker state used to choose where temporal chunks should be
/// prefetched. Counts come from the router index and are intentionally simple:
/// they describe resident chunks and currently inflight prefetch chunks, not a
/// second routing policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkerPrefetchCandidate {
    pub worker: WorkerWithDpRank,
    pub resident_chunks: u32,
    pub prefetch_inflight_chunks: u32,
}

/// The result of one route-time global-prefetch decision. Reservations are
/// installed in Dynamo before the network hint is sent, so the next route can
/// see them immediately.
#[derive(Debug, Clone, Default)]
pub(crate) struct GlobalPrefetchDispatch {
    /// Workers carrying request-local warmup work that may be cancelled when
    /// the demand request finishes. Temporal association work is deliberately
    /// not included: it outlives the triggering request.
    pub cancel_workers: Vec<WorkerWithDpRank>,
    pub reservations: Vec<PrefetchReservation>,
}

#[derive(Debug, Clone)]
pub(crate) struct PrefetchReservation {
    pub worker: WorkerWithDpRank,
    pub block_hashes: Vec<LocalBlockHash>,
}

/// Select the worker with the lowest estimated completion time:
///
/// `score = queue_delay + missing_copy_time`
///
/// Both terms are expressed in seconds using the same estimated chunk
/// throughput. No route affinity, load, or routing score is included here.
pub(crate) fn select_prefetch_worker(
    candidates: &[WorkerPrefetchCandidate],
    target_chunks: usize,
    chunks_per_second: f64,
    fallback: WorkerWithDpRank,
) -> WorkerWithDpRank {
    if candidates.is_empty() || target_chunks == 0 {
        return fallback;
    }
    let throughput = if chunks_per_second.is_finite() && chunks_per_second > 0.0 {
        chunks_per_second
    } else {
        1.0
    };
    candidates
        .iter()
        .copied()
        .map(|candidate| {
            let resident = candidate.resident_chunks as usize;
            let missing = target_chunks.saturating_sub(resident);
            let score = (candidate.prefetch_inflight_chunks as f64 + missing as f64) / throughput;
            (candidate, score)
        })
        .min_by(|(a, a_score), (b, b_score)| {
            a_score
                .total_cmp(b_score)
                .then_with(|| a.worker.cmp(&b.worker))
        })
        .map(|(candidate, _)| candidate.worker)
        .unwrap_or(fallback)
}

#[derive(Debug, Clone)]
struct ChunkDescriptor {
    /// The complete prefix ending at this chunk.  LMCache's chunk key is a
    /// prefix hash, not a hash of the chunk in isolation.  Keeping one shared
    /// allocation per observed request lets the temporal dispatcher send the
    /// exact prefix while selecting only this chunk with start/end_chunk.
    prefix_tokens: Arc<[u32]>,
    start_chunk: u32,
    end_chunk: u32,
}

#[derive(Debug, Clone)]
struct SessionState {
    model_id: String,
    last_chunks: Vec<ChunkKey>,
    last_seen: Instant,
}

#[derive(Debug, Clone)]
struct TargetStats {
    support: f64,
    last_seen: Instant,
}

#[derive(Debug, Clone)]
struct SourceStats {
    total: f64,
    last_seen: Instant,
    targets: HashMap<ChunkKey, TargetStats>,
}

#[derive(Debug, Default)]
struct ManagerState {
    sessions: HashMap<String, SessionState>,
    catalog: HashMap<ChunkKey, ChunkDescriptor>,
    catalog_order: VecDeque<ChunkKey>,
    sources: HashMap<ChunkKey, SourceStats>,
    // Active physical attempts. A failure callback removes the matching
    // timestamp immediately; TTL remains only as a safety net for lost
    // completion notifications.
    inflight: HashMap<(ChunkKey, WorkerWithDpRank), Instant>,
    /// Distinct sessions in which each chunk has been observed.  This is used
    /// only by the optional stable-prefix filter and is bounded with the
    /// chunk catalog.
    chunk_sessions: HashMap<ChunkKey, HashSet<String>>,
    scheduled: u64,
    predicted: u64,
    dropped: u64,
}

/// A bounded router-owned chunk predictor and dispatcher.
pub(crate) struct GlobalPrefetchManager {
    config: GlobalPrefetchConfig,
    state: Mutex<ManagerState>,
    dispatcher: Arc<PrefetchHintDispatcher>,
}

impl GlobalPrefetchManager {
    pub(crate) fn from_env(dispatcher: Option<Arc<PrefetchHintDispatcher>>) -> Option<Arc<Self>> {
        let config = GlobalPrefetchConfig::from_env();
        if !config.enabled {
            return None;
        }
        let Some(dispatcher) = dispatcher else {
            tracing::warn!(
                "DYN_GLOBAL_PREFETCH_ENABLED requires the LMCache prefetch hint dispatcher"
            );
            return None;
        };
        tracing::info!(
            chunk_size = config.chunk_size,
            top_k = config.top_k,
            min_support = config.min_support,
            min_probability = config.min_probability,
            filter_stable_chunks = config.filter_stable_chunks,
            stable_session_threshold = config.stable_session_threshold,
            "Global Dynamo chunk prefetch manager enabled"
        );
        Some(Arc::new(Self {
            config,
            state: Mutex::new(ManagerState::default()),
            dispatcher,
        }))
    }

    /// Observe a committed route and issue current-request and chunk-temporal
    /// hints. Network work is spawned by `PrefetchHintDispatcher`.
    pub(crate) fn on_route_committed(
        self: &Arc<Self>,
        request_id: &str,
        session_id: Option<&str>,
        model_id: &str,
        tokens: &[u32],
        worker: WorkerWithDpRank,
        candidates: &[WorkerPrefetchCandidate],
        route_epoch: u64,
        deadline_ns: Option<u64>,
        max_prefetch_bytes: Option<u64>,
        reserve_callback: impl FnOnce(&[PrefetchReservation]),
        release_callback: Arc<dyn Fn(&[PrefetchReservation]) + Send + Sync + 'static>,
    ) -> GlobalPrefetchDispatch {
        let normalized_len = tokens.len() / self.config.chunk_size * self.config.chunk_size;
        if normalized_len == 0 {
            return GlobalPrefetchDispatch::default();
        }
        let normalized = &tokens[..normalized_len];
        let chunks = chunk_descriptors(model_id, normalized, self.config.chunk_size);
        if chunks.is_empty() {
            return GlobalPrefetchDispatch::default();
        }
        let current_chunks: Vec<ChunkKey> = chunks.iter().map(|(key, _)| key.clone()).collect();
        let current_set: HashSet<ChunkKey> = current_chunks.iter().cloned().collect();
        let now = Instant::now();

        let temporal_targets = {
            let mut state = self.state.lock().expect("global prefetch state poisoned");
            self.expire_locked(&mut state, now);
            self.stage_chunks_locked(&mut state, chunks, session_id);

            let previous = session_id.and_then(|session| state.sessions.get(session).cloned());
            let previous_chunks = previous
                .as_ref()
                .map(|previous| self.filter_stable_chunks(&state, &previous.last_chunks))
                .unwrap_or_default();
            let learning_current_chunks = self.filter_stable_chunks(&state, &current_chunks);
            let previous_matches = previous.as_ref().is_some_and(|previous| {
                previous.model_id == model_id
                    && now.duration_since(previous.last_seen) <= self.config.session_ttl
            });
            let targets = previous
                .as_ref()
                .filter(|_| previous_matches)
                .map(|_| self.best_candidates_locked(&state, &previous_chunks, &current_set, now))
                .unwrap_or_default();

            if let Some(session) = session_id {
                if previous_matches {
                    self.observe_transition_locked(
                        &mut state,
                        &previous_chunks,
                        &learning_current_chunks,
                        now,
                    );
                }
                state.sessions.insert(
                    session.to_string(),
                    SessionState {
                        model_id: model_id.to_string(),
                        last_chunks: current_chunks
                            .iter()
                            .take(self.config.max_observation_chunks)
                            .cloned()
                            .collect(),
                        last_seen: now,
                    },
                );
                self.trim_sessions_locked(&mut state);
            }
            targets
        };

        tracing::info!(
            request_id,
            session_id = ?session_id,
            current_chunks = current_chunks.len(),
            temporal_targets = temporal_targets.len(),
            "Global prefetch association observation"
        );

        let temporal_worker = select_prefetch_worker(
            candidates,
            temporal_targets.len(),
            self.config.chunks_per_second,
            worker,
        );
        let has_temporal_targets = !temporal_targets.is_empty();
        if has_temporal_targets {
            tracing::debug!(
                request_id,
                current_worker = ?worker,
                selected_prefetch_worker = ?temporal_worker,
                predicted_chunks = temporal_targets.len(),
                "Selected temporal prefetch target using queue-plus-copy score"
            );
        }
        let mut cancel_workers = Vec::new();
        let mut reservations = Vec::new();
        let mut temporal_dispatches = Vec::new();

        if self.config.temporal_enabled {
            let mut reserved_hashes = Vec::new();
            for target in temporal_targets {
                let descriptor = self
                    .state
                    .lock()
                    .ok()
                    .and_then(|state| state.catalog.get(&target).cloned());
                let Some(descriptor) = descriptor else {
                    continue;
                };
                let target_hash = target.1;
                let Some(inflight_started) = self.reserve_inflight(target, temporal_worker) else {
                    continue;
                };
                reserved_hashes.push(LocalBlockHash(target_hash));
                temporal_dispatches.push((
                    target_hash,
                    descriptor,
                    PrefetchReservation {
                        worker: temporal_worker,
                        block_hashes: vec![LocalBlockHash(target_hash)],
                    },
                    inflight_started,
                ));
            }
            if !reserved_hashes.is_empty() {
                reservations.extend(
                    temporal_dispatches
                        .iter()
                        .map(|(_, _, reservation, _)| reservation.clone()),
                );
            }
        }

        // Install the router-local reservation before any asynchronous network
        // work is spawned below. This makes the next route's scheduler view
        // deterministic even when the LMCache event arrives later.
        reserve_callback(&reservations);

        if self.config.current_request_enabled {
            // This is a deterministic warmup of the current request. The
            // association predictor below dispatches individual chunks.
            self.dispatcher.dispatch_with_options(
                normalized.to_vec(),
                Some(request_id.to_string()),
                session_id.map(str::to_string),
                Some(model_id.to_string()),
                Some(format!("global-current-{request_id}")),
                worker,
                route_epoch,
                deadline_ns,
                max_prefetch_bytes,
                Some(0),
                None,
                self.config.ttl.as_millis().min(u64::MAX as u128) as u64,
            );
            cancel_workers.push(worker);
        }

        if self.config.temporal_enabled {
            for (target_hash, descriptor, reservation, inflight_started) in temporal_dispatches {
                let release_callback = Arc::clone(&release_callback);
                // `inflight` is an active data-plane reservation, not a
                // residency record.  If LMCache reports a physical failure,
                // clear it immediately so a later association can retry the
                // chunk instead of waiting for the (possibly long) TTL.
                let manager = Arc::clone(self);
                let inflight_target = (model_id.to_string(), target_hash);
                let on_failure: PrefetchFailureCallback = Arc::new(move || {
                    release_callback(std::slice::from_ref(&reservation));
                    manager.release_inflight(&inflight_target, temporal_worker, inflight_started);
                });
                self.dispatcher.dispatch_with_options_monitored(
                    descriptor.prefix_tokens.to_vec(),
                    Some(request_id.to_string()),
                    session_id.map(str::to_string),
                    Some(model_id.to_string()),
                    Some(format!(
                        "global-temporal-chunk-{request_id}-{:016x}",
                        target_hash
                    )),
                    temporal_worker,
                    route_epoch,
                    deadline_ns,
                    max_prefetch_bytes,
                    Some(descriptor.start_chunk),
                    Some(descriptor.end_chunk),
                    self.config.ttl.as_millis().min(u64::MAX as u128) as u64,
                    Some(on_failure),
                );
            }
        }
        cancel_workers.sort_unstable();
        cancel_workers.dedup();
        GlobalPrefetchDispatch {
            cancel_workers,
            reservations,
        }
    }

    fn stage_chunks_locked(
        &self,
        state: &mut ManagerState,
        chunks: Vec<(ChunkKey, ChunkDescriptor)>,
        session_id: Option<&str>,
    ) {
        for (chunk_id, descriptor) in chunks {
            if self.config.filter_stable_chunks {
                if let Some(session_id) = session_id {
                    record_chunk_session(
                        state,
                        &chunk_id,
                        session_id,
                        self.config.stable_session_threshold,
                    );
                }
            }
            state.catalog.insert(chunk_id.clone(), descriptor);
            state.catalog_order.retain(|id| *id != chunk_id);
            state.catalog_order.push_back(chunk_id);
        }
        while state.catalog_order.len() > self.config.max_chunk_catalog {
            if let Some(old) = state.catalog_order.pop_front() {
                state.catalog.remove(&old);
                state.chunk_sessions.remove(&old);
            }
        }
    }

    fn filter_stable_chunks(&self, state: &ManagerState, chunks: &[ChunkKey]) -> Vec<ChunkKey> {
        if !self.config.filter_stable_chunks {
            return chunks
                .iter()
                .take(self.config.max_observation_chunks)
                .cloned()
                .collect();
        }
        chunks
            .iter()
            .take(self.config.max_observation_chunks)
            .filter(|chunk| !self.chunk_is_stable(state, chunk))
            .cloned()
            .collect()
    }

    fn chunk_is_stable(&self, state: &ManagerState, chunk: &ChunkKey) -> bool {
        self.config.filter_stable_chunks
            && chunk_is_stable(state, chunk, self.config.stable_session_threshold)
    }

    fn observe_transition_locked(
        &self,
        state: &mut ManagerState,
        previous_chunks: &[ChunkKey],
        current_chunks: &[ChunkKey],
        now: Instant,
    ) {
        // A repeated local chunk hash within one request is still one
        // observation of that chunk.  Deduplicate both sides before updating
        // the association counts so a repeated token pattern cannot make a
        // probability exceed one.
        let current = unique_chunk_refs(current_chunks, self.config.max_observation_chunks);
        let sources = unique_chunk_refs(previous_chunks, self.config.max_observation_chunks);
        for source in sources {
            let targets: Vec<&ChunkKey> = current
                .iter()
                .copied()
                .filter(|target| *target != source)
                .collect();
            if targets.is_empty() {
                continue;
            }
            let source_stats = state
                .sources
                .entry(source.clone())
                .or_insert_with(|| SourceStats {
                    total: 0.0,
                    last_seen: now,
                    targets: HashMap::new(),
                });
            record_source_targets(source_stats, &targets, now, self.config.half_life);
            self.trim_targets(source_stats, now);
        }
        while state.sources.len() > self.config.max_chunk_catalog {
            let Some(oldest) = state
                .sources
                .iter()
                .min_by_key(|(_, stats)| stats.last_seen)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            state.sources.remove(&oldest);
        }
    }

    fn trim_targets(&self, source: &mut SourceStats, now: Instant) {
        if source.targets.len() <= self.config.top_k * 2 {
            return;
        }
        let mut targets: Vec<ChunkKey> = source.targets.keys().cloned().collect();
        targets.sort_by(|a, b| {
            let a_score = decay(
                source.targets[a].support,
                source.targets[a].last_seen,
                now,
                self.config.half_life,
            );
            let b_score = decay(
                source.targets[b].support,
                source.targets[b].last_seen,
                now,
                self.config.half_life,
            );
            b_score.total_cmp(&a_score).then_with(|| a.cmp(b))
        });
        targets.truncate(self.config.top_k);
        source.targets.retain(|key, _| targets.contains(key));
    }

    fn best_candidates_locked(
        &self,
        state: &ManagerState,
        previous_chunks: &[ChunkKey],
        current_chunks: &HashSet<ChunkKey>,
        now: Instant,
    ) -> Vec<ChunkKey> {
        let mut best: HashMap<ChunkKey, f64> = HashMap::new();
        for source in previous_chunks
            .iter()
            .take(self.config.max_observation_chunks)
        {
            let Some(source_stats) = state.sources.get(source) else {
                continue;
            };
            let total = decay(
                source_stats.total,
                source_stats.last_seen,
                now,
                self.config.half_life,
            );
            if total < self.config.min_support {
                continue;
            }
            for (target, target_stats) in &source_stats.targets {
                if current_chunks.contains(target) {
                    continue;
                }
                if self.chunk_is_stable(state, target) {
                    continue;
                }
                let support = decay(
                    target_stats.support,
                    target_stats.last_seen,
                    now,
                    self.config.half_life,
                );
                let probability = support / total;
                if support >= self.config.min_support
                    && probability >= self.config.min_probability
                    && state.catalog.contains_key(target)
                {
                    best.entry(target.clone())
                        .and_modify(|score| *score = score.max(probability))
                        .or_insert(probability);
                }
            }
        }
        let mut candidates: Vec<(ChunkKey, f64)> = best.into_iter().collect();
        // HashMap iteration order is intentionally unstable.  A stable
        // tie-breaker is important here because all chunks from one strongly
        // associated request often have the same probability.
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        candidates
            .into_iter()
            .take(self.config.top_k)
            .map(|(target, _)| target)
            .collect()
    }

    fn reserve_inflight(&self, target: ChunkKey, worker: WorkerWithDpRank) -> Option<Instant> {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(_) => return None,
        };
        if state.inflight.len() >= self.config.max_inflight
            || state.inflight.contains_key(&(target.clone(), worker))
        {
            state.dropped += 1;
            return None;
        }
        let started = Instant::now();
        state.inflight.insert((target, worker), started);
        state.scheduled += 1;
        state.predicted += 1;
        Some(started)
    }

    fn release_inflight(&self, target: &ChunkKey, worker: WorkerWithDpRank, started: Instant) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if release_inflight_locked(&mut state, target, worker, started) {
            tracing::debug!(
                worker = ?worker,
                target_hash = format_args!("{:016x}", target.1),
                "Released failed global prefetch inflight reservation"
            );
        }
    }

    fn expire_locked(&self, state: &mut ManagerState, now: Instant) {
        state
            .sessions
            .retain(|_, value| now.duration_since(value.last_seen) <= self.config.session_ttl);
        state
            .inflight
            .retain(|_, started| now.duration_since(*started) <= self.config.ttl);
    }

    fn trim_sessions_locked(&self, state: &mut ManagerState) {
        while state.sessions.len() > self.config.max_sessions {
            let oldest = state
                .sessions
                .iter()
                .min_by_key(|(_, state)| state.last_seen)
                .map(|(id, _)| id.clone());
            if let Some(oldest) = oldest {
                state.sessions.remove(&oldest);
            } else {
                break;
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn stats(&self) -> (u64, u64, u64, usize, usize) {
        self.state
            .lock()
            .map(|state| {
                (
                    state.scheduled,
                    state.predicted,
                    state.dropped,
                    state.sessions.len(),
                    state.catalog.len(),
                )
            })
            .unwrap_or_default()
    }
}

fn release_inflight_locked(
    state: &mut ManagerState,
    target: &ChunkKey,
    worker: WorkerWithDpRank,
    started: Instant,
) -> bool {
    let key = (target.clone(), worker);
    if state.inflight.get(&key).copied() != Some(started) {
        return false;
    }
    state.inflight.remove(&key).is_some()
}

fn chunk_descriptors(
    model_id: &str,
    tokens: &[u32],
    chunk_size: usize,
) -> Vec<(ChunkKey, ChunkDescriptor)> {
    let block_hashes = compute_block_hash_for_seq(tokens, chunk_size as u32, None);
    // Every descriptor shares the request's prefix allocation.  Temporal
    // hints need the complete prefix so LMCache can reproduce its own
    // parent-dependent cache key; start_chunk/end_chunk keep the operation
    // bounded to the predicted chunk.
    let prefix_tokens: Arc<[u32]> = Arc::from(tokens.to_vec().into_boxed_slice());
    tokens
        .chunks_exact(chunk_size)
        .zip(block_hashes)
        .enumerate()
        .map(|(chunk_index, (_chunk, block_hash))| {
            (
                (model_id.to_string(), block_hash.0),
                ChunkDescriptor {
                    prefix_tokens: Arc::clone(&prefix_tokens),
                    start_chunk: chunk_index as u32,
                    end_chunk: chunk_index.saturating_add(1) as u32,
                },
            )
        })
        .collect()
}

fn unique_chunk_refs<'a>(chunks: &'a [ChunkKey], limit: usize) -> Vec<&'a ChunkKey> {
    let mut unique = Vec::with_capacity(limit.min(chunks.len()));
    for chunk in chunks.iter().take(limit) {
        if !unique.iter().any(|existing| *existing == chunk) {
            unique.push(chunk);
        }
    }
    unique
}

fn record_source_targets(
    source_stats: &mut SourceStats,
    targets: &[&ChunkKey],
    now: Instant,
    half_life: Duration,
) {
    // `total` is the denominator of P(target | source): one source
    // observation contributes one transition, independent of how many
    // target chunks happen to be present in that request.  The old code
    // added `targets.len()` here, which made a perfectly deterministic
    // multi-chunk successor look unlikely and forced callers to lower the
    // probability threshold until noisy candidates passed.
    source_stats.total = decay(source_stats.total, source_stats.last_seen, now, half_life) + 1.0;
    source_stats.last_seen = now;
    for target in targets {
        let target_stats = source_stats
            .targets
            .entry((*target).clone())
            .or_insert(TargetStats {
                support: 0.0,
                last_seen: now,
            });
        target_stats.support =
            decay(target_stats.support, target_stats.last_seen, now, half_life) + 1.0;
        target_stats.last_seen = now;
    }
}

fn decay(value: f64, last_seen: Instant, now: Instant, half_life: Duration) -> f64 {
    if value <= 0.0 {
        return 0.0;
    }
    let age = now.duration_since(last_seen).as_secs_f64();
    value * (-std::f64::consts::LN_2 * age / half_life.as_secs_f64()).exp()
}

fn record_chunk_session(
    state: &mut ManagerState,
    chunk: &ChunkKey,
    session_id: &str,
    threshold: usize,
) {
    let sessions = state.chunk_sessions.entry(chunk.clone()).or_default();
    // Once the threshold is reached, more identities cannot change the
    // stable/not-stable decision. Keep only threshold identities so the
    // predictor's metadata remains bounded even with many sessions.
    if sessions.len() < threshold {
        sessions.insert(session_id.to_string());
    }
}

fn chunk_is_stable(state: &ManagerState, chunk: &ChunkKey, threshold: usize) -> bool {
    state
        .chunk_sessions
        .get(chunk)
        .map(|sessions| sessions.len() >= threshold)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::{
        GlobalPrefetchConfig, ManagerState, SourceStats, TargetStats, WorkerPrefetchCandidate,
        chunk_descriptors, chunk_is_stable, decay, record_chunk_session, record_source_targets,
        release_inflight_locked, select_prefetch_worker,
    };
    use crate::kv_router::protocols::WorkerWithDpRank;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn chunk_identity_does_not_include_parent_sequence() {
        let first = vec![1u32; 4];
        let mut second_request = first.clone();
        second_request.extend([2u32; 4]);
        let first_chunks = chunk_descriptors("model", &first, 4);
        let second_chunks = chunk_descriptors("model", &second_request, 4);
        assert_eq!(first_chunks[0].0, second_chunks[0].0);
        assert_ne!(first_chunks[0].0, second_chunks[1].0);
    }

    #[test]
    fn chunk_descriptor_contains_prefix_and_target_range() {
        let chunks = chunk_descriptors("model", &[1, 2, 3, 4, 5, 6], 2);
        assert_eq!(&*chunks[0].1.prefix_tokens, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(&*chunks[1].1.prefix_tokens, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(&*chunks[2].1.prefix_tokens, &[1, 2, 3, 4, 5, 6]);
        assert_eq!((chunks[0].1.start_chunk, chunks[0].1.end_chunk), (0, 1));
        assert_eq!((chunks[1].1.start_chunk, chunks[1].1.end_chunk), (1, 2));
        assert_eq!((chunks[2].1.start_chunk, chunks[2].1.end_chunk), (2, 3));
    }

    #[test]
    fn descriptors_share_prefix_storage() {
        let chunks = chunk_descriptors("model", &[1, 2, 3, 4], 2);
        assert!(Arc::ptr_eq(
            &chunks[0].1.prefix_tokens,
            &chunks[1].1.prefix_tokens
        ));
    }

    #[test]
    fn decay_reduces_old_support() {
        let now = Instant::now();
        let value = decay(
            2.0,
            now - Duration::from_secs(10),
            now,
            Duration::from_secs(10),
        );
        assert!((value - 1.0).abs() < 0.01);
    }

    #[test]
    fn association_probability_uses_transition_count_not_target_count() {
        let target_b = ("model".to_string(), 2);
        let target_c = ("model".to_string(), 3);
        let now = Instant::now();
        let half_life = Duration::from_secs(300);
        let mut stats = SourceStats {
            total: 0.0,
            last_seen: now,
            targets: std::collections::HashMap::from([
                (
                    target_b.clone(),
                    TargetStats {
                        support: 0.0,
                        last_seen: now,
                    },
                ),
                (
                    target_c.clone(),
                    TargetStats {
                        support: 0.0,
                        last_seen: now,
                    },
                ),
            ]),
        };

        record_source_targets(&mut stats, &[&target_b, &target_c], now, half_life);
        assert_eq!(stats.total, 1.0);
        assert_eq!(stats.targets[&target_b].support, 1.0);
        assert_eq!(stats.targets[&target_c].support, 1.0);

        // B is present in both transitions while C is present only in the
        // first: P(B|source)=1 and P(C|source)=1/2.
        record_source_targets(&mut stats, &[&target_b], now, half_life);
        assert!((stats.targets[&target_b].support / stats.total - 1.0).abs() < 1e-9);
        assert!((stats.targets[&target_c].support / stats.total - 0.5).abs() < 1e-9);
    }

    #[test]
    fn defaults_are_bounded() {
        let config = GlobalPrefetchConfig::from_env();
        assert!(config.chunk_size > 0);
        assert!(config.max_observation_chunks > 0);
        assert!(config.max_inflight > 0);
        assert!(config.chunks_per_second > 0.0);
    }

    #[test]
    fn stable_chunk_counts_distinct_sessions_and_caps_metadata() {
        let chunk = ("model".to_string(), 7);
        let mut state = ManagerState::default();
        for session in ["s1", "s2", "s3"] {
            record_chunk_session(&mut state, &chunk, session, 4);
        }
        assert!(!chunk_is_stable(&state, &chunk, 4));

        // Re-observing one session is not a new distinct-session observation.
        record_chunk_session(&mut state, &chunk, "s1", 4);
        assert_eq!(state.chunk_sessions[&chunk].len(), 3);

        record_chunk_session(&mut state, &chunk, "s4", 4);
        assert!(chunk_is_stable(&state, &chunk, 4));
        record_chunk_session(&mut state, &chunk, "s5", 4);
        assert_eq!(state.chunk_sessions[&chunk].len(), 4);
    }

    #[test]
    fn physical_failure_releases_inflight_reservation_for_retry() {
        let worker = WorkerWithDpRank::new(7, 0);
        let target = ("model".to_string(), 0xdead_beef);
        let mut state = ManagerState::default();
        let started = Instant::now();
        state.inflight.insert((target.clone(), worker), started);

        assert!(release_inflight_locked(
            &mut state, &target, worker, started
        ));
        assert!(!state.inflight.contains_key(&(target.clone(), worker)));
        assert!(!release_inflight_locked(
            &mut state, &target, worker, started
        ));
        // The next association observation can reserve the same target again.
        let restarted = Instant::now();
        assert!(
            state
                .inflight
                .insert((target.clone(), worker), restarted)
                .is_none()
        );
        // A late failure from the previous task must not clear this newer
        // reservation.
        assert!(!release_inflight_locked(
            &mut state, &target, worker, started
        ));
        assert!(release_inflight_locked(
            &mut state, &target, worker, restarted
        ));
    }

    #[test]
    fn target_score_balances_queue_and_copy_work() {
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);
        let candidates = [
            WorkerPrefetchCandidate {
                worker: worker_a,
                resident_chunks: 3,
                prefetch_inflight_chunks: 8,
            },
            WorkerPrefetchCandidate {
                worker: worker_b,
                resident_chunks: 0,
                prefetch_inflight_chunks: 0,
            },
        ];
        // A needs one copy but has eight chunks queued; B needs four copies.
        // With four chunks/second, B completes sooner (1s vs 2.25s).
        assert_eq!(
            select_prefetch_worker(&candidates, 4, 4.0, worker_a),
            worker_b
        );
    }

    #[test]
    fn target_score_uses_resident_chunks() {
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);
        let candidates = [
            WorkerPrefetchCandidate {
                worker: worker_a,
                resident_chunks: 4,
                prefetch_inflight_chunks: 0,
            },
            WorkerPrefetchCandidate {
                worker: worker_b,
                resident_chunks: 0,
                prefetch_inflight_chunks: 0,
            },
        ];
        assert_eq!(
            select_prefetch_worker(&candidates, 4, 4.0, worker_b),
            worker_a
        );
    }
}
