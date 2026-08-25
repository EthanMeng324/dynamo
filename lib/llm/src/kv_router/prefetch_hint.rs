// SPDX-License-Identifier: Apache-2.0
//! Non-blocking Dynamo -> LMCache route-time prefetch hints.

use std::{collections::HashMap, env, sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};

use super::protocols::WorkerWithDpRank;

#[derive(Debug, Clone)]
struct PrefetchHintConfig {
    enabled: bool,
    endpoint: Option<String>,
    instance_template: Option<String>,
    instance_map: HashMap<u64, String>,
    max_bytes: Option<u64>,
    priority: i32,
    timeout: Duration,
}

impl PrefetchHintConfig {
    fn from_env() -> Self {
        Self {
            enabled: env_bool("DYN_PREFETCH_HINTS_ENABLED", false)
                || env_bool("DYN_GLOBAL_PREFETCH_ENABLED", false),
            endpoint: env::var("DYN_PREFETCH_HINT_ENDPOINT").ok(),
            instance_template: env::var("DYN_PREFETCH_HINT_INSTANCE_TEMPLATE").ok(),
            instance_map: parse_instance_map(),
            max_bytes: env::var("DYN_PREFETCH_HINT_MAX_BYTES")
                .ok()
                .and_then(|value| value.parse().ok()),
            priority: env::var("DYN_PREFETCH_HINT_PRIORITY")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
            timeout: Duration::from_micros(
                env::var("DYN_PREFETCH_HINT_TIMEOUT_US")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(5000),
            ),
        }
    }
}

fn parse_instance_map() -> HashMap<u64, String> {
    parse_instance_map_value(&env::var("DYN_PREFETCH_HINT_INSTANCE_MAP").unwrap_or_default())
}

fn parse_instance_map_value(value: &str) -> HashMap<u64, String> {
    value
        .split(',')
        .filter_map(|entry| {
            let (worker_id, instance_id) = entry.split_once(['=', ':'])?;
            let worker_id = worker_id.trim().parse::<u64>().ok()?;
            let instance_id = instance_id.trim();
            (!instance_id.is_empty()).then(|| (worker_id, instance_id.to_owned()))
        })
        .collect()
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<bool>().ok())
        .unwrap_or(default)
}

#[derive(Debug, Serialize)]
struct HintRequest {
    instance_id: String,
    tokens: Vec<u32>,
    request_id: Option<String>,
    session_id: Option<String>,
    model_id: Option<String>,
    task_id: Option<String>,
    route_epoch: u64,
    deadline_ns: Option<u64>,
    max_prefetch_bytes: Option<u64>,
    priority: i32,
    start_chunk: Option<u32>,
    end_chunk: Option<u32>,
    ttl_ms: u64,
}

#[derive(Debug, Deserialize)]
struct HintResponse {
    #[serde(default)]
    accepted: usize,
    #[serde(default)]
    scheduled: usize,
    #[serde(default)]
    stale: bool,
}

#[derive(Debug, Deserialize)]
struct PrefetchStatusResponse {
    #[serde(default)]
    state: String,
}

pub(crate) type PrefetchFailureCallback = Arc<dyn Fn() + Send + Sync + 'static>;

#[derive(Debug, Serialize)]
struct CancelRequest {
    instance_id: String,
    request_id: String,
    route_epoch: u64,
}

pub(crate) struct PrefetchHintDispatcher {
    config: PrefetchHintConfig,
    client: reqwest::Client,
}

#[cfg(test)]
mod tests {
    use super::parse_instance_map_value;

    #[test]
    fn parses_worker_to_instance_map_and_ignores_invalid_entries() {
        let parsed = parse_instance_map_value(
            "101=global-pf-node0, 202:global-pf-node1,invalid, :empty,not-a-number=x",
        );
        assert_eq!(
            parsed.get(&101).map(String::as_str),
            Some("global-pf-node0")
        );
        assert_eq!(
            parsed.get(&202).map(String::as_str),
            Some("global-pf-node1")
        );
        assert_eq!(parsed.len(), 2);
    }
}

impl PrefetchHintDispatcher {
    pub(crate) fn from_env() -> Option<Arc<Self>> {
        let config = PrefetchHintConfig::from_env();
        if !config.enabled {
            return None;
        }
        if config.endpoint.is_none()
            || (config.instance_template.is_none() && config.instance_map.is_empty())
        {
            tracing::warn!(
                "DYN_PREFETCH_HINTS_ENABLED requires DYN_PREFETCH_HINT_ENDPOINT and \
                 either DYN_PREFETCH_HINT_INSTANCE_TEMPLATE or \
                 DYN_PREFETCH_HINT_INSTANCE_MAP"
            );
            return None;
        }
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("valid prefetch hint client configuration");
        tracing::info!(
            endpoint = %config.endpoint.as_deref().unwrap_or(""),
            instance_template = ?config.instance_template,
            instance_map_entries = config.instance_map.len(),
            "Route-time LMCache prefetch hints enabled"
        );
        Some(Arc::new(Self { config, client }))
    }

    fn instance_id(&self, worker: WorkerWithDpRank) -> Option<String> {
        if let Some(instance_id) = self.config.instance_map.get(&worker.worker_id) {
            return Some(instance_id.clone());
        }
        self.config.instance_template.as_ref().map(|template| {
            template
                .replace("{worker_id}", &worker.worker_id.to_string())
                .replace("{dp_rank}", &worker.dp_rank.to_string())
        })
    }

    pub(crate) fn dispatch(
        self: &Arc<Self>,
        tokens: Vec<u32>,
        request_id: Option<String>,
        worker: WorkerWithDpRank,
        route_epoch: u64,
        deadline_ns: Option<u64>,
    ) {
        self.dispatch_with_options(
            tokens,
            request_id,
            None,
            None,
            None,
            worker,
            route_epoch,
            deadline_ns,
            None,
            None,
            None,
            5000,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_with_options(
        self: &Arc<Self>,
        tokens: Vec<u32>,
        request_id: Option<String>,
        session_id: Option<String>,
        model_id: Option<String>,
        task_id: Option<String>,
        worker: WorkerWithDpRank,
        route_epoch: u64,
        deadline_ns: Option<u64>,
        max_prefetch_bytes: Option<u64>,
        start_chunk: Option<u32>,
        end_chunk: Option<u32>,
        ttl_ms: u64,
    ) {
        self.dispatch_with_options_monitored(
            tokens,
            request_id,
            session_id,
            model_id,
            task_id,
            worker,
            route_epoch,
            deadline_ns,
            max_prefetch_bytes,
            start_chunk,
            end_chunk,
            ttl_ms,
            None,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_with_options_monitored(
        self: &Arc<Self>,
        tokens: Vec<u32>,
        request_id: Option<String>,
        session_id: Option<String>,
        model_id: Option<String>,
        task_id: Option<String>,
        worker: WorkerWithDpRank,
        route_epoch: u64,
        deadline_ns: Option<u64>,
        max_prefetch_bytes: Option<u64>,
        start_chunk: Option<u32>,
        end_chunk: Option<u32>,
        ttl_ms: u64,
        on_failure: Option<PrefetchFailureCallback>,
    ) {
        // A route-time hint without prompt tokens cannot identify a cache
        // prefix.  Do not send it to LMCache: the controller intentionally
        // rejects empty token lists with HTTP 422.
        if tokens.is_empty() {
            tracing::debug!("Skipping empty LMCache prefetch hint");
            if let Some(on_failure) = on_failure {
                on_failure();
            }
            return;
        }
        let Some(endpoint) = self.config.endpoint.clone() else {
            if let Some(on_failure) = on_failure {
                on_failure();
            }
            return;
        };
        let Some(instance_id) = self.instance_id(worker) else {
            if let Some(on_failure) = on_failure {
                on_failure();
            }
            return;
        };
        let task_id_for_status = task_id.clone();
        let instance_id_for_status = instance_id.clone();
        let request = HintRequest {
            instance_id,
            tokens,
            request_id,
            session_id,
            model_id,
            task_id,
            route_epoch,
            deadline_ns,
            max_prefetch_bytes: max_prefetch_bytes.or(self.config.max_bytes),
            priority: self.config.priority,
            start_chunk,
            end_chunk,
            ttl_ms,
        };
        let dispatcher = Arc::clone(self);
        tokio::spawn(async move {
            match dispatcher.client.post(endpoint).json(&request).send().await {
                Ok(response) => {
                    let status = response.status();
                    match response.text().await {
                        Ok(body) if status.is_success() => {
                            match serde_json::from_str::<HintResponse>(&body) {
                                Ok(body) => {
                                    tracing::debug!(
                                        accepted = body.accepted,
                                        scheduled = body.scheduled,
                                        stale = body.stale,
                                        worker_id = worker.worker_id,
                                        "Route-time LMCache prefetch hint completed"
                                    );
                                    if body.stale || body.scheduled == 0 {
                                        if let Some(on_failure) = on_failure.as_ref() {
                                            on_failure();
                                        }
                                    } else if let Some(on_failure) = on_failure {
                                        if let Some(task_id) = task_id_for_status.as_deref() {
                                            dispatcher.monitor_task(
                                                instance_id_for_status,
                                                task_id.to_owned(),
                                                on_failure,
                                            );
                                        } else {
                                            // A successful response without a task id cannot be
                                            // associated with a physical completion state.
                                            on_failure();
                                        }
                                    }
                                }
                                Err(error) => {
                                    tracing::debug!(%error, response = %body, "Invalid LMCache prefetch hint response");
                                    if let Some(on_failure) = on_failure.as_ref() {
                                        on_failure();
                                    }
                                }
                            }
                        }
                        Ok(body) => {
                            tracing::warn!(
                                status = %status,
                                worker_id = worker.worker_id,
                                response = %body,
                                "LMCache prefetch hint rejected"
                            );
                            if let Some(on_failure) = on_failure.as_ref() {
                                on_failure();
                            }
                        }
                        Err(error) => {
                            tracing::debug!(%error, status = %status, "Failed to read LMCache prefetch hint response");
                            if let Some(on_failure) = on_failure.as_ref() {
                                on_failure();
                            }
                        }
                    }
                }
                Err(error) => {
                    tracing::debug!(%error, "LMCache prefetch hint dispatch failed");
                    if let Some(on_failure) = on_failure.as_ref() {
                        on_failure();
                    }
                }
            }
        });
    }

    fn monitor_task(
        self: &Arc<Self>,
        instance_id: String,
        task_id: String,
        on_failure: PrefetchFailureCallback,
    ) {
        let Some(endpoint) = self.config.endpoint.clone() else {
            on_failure();
            return;
        };
        let status_endpoint = if endpoint.ends_with("/v1/cxl/prefetch") {
            format!("{endpoint}/{task_id}")
        } else if endpoint.ends_with("/prefetch_hint") {
            format!("{endpoint}/{task_id}")
        } else {
            format!("{endpoint}/v1/cxl/prefetch/{task_id}")
        };
        let dispatcher = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                match dispatcher
                    .client
                    .get(&status_endpoint)
                    .query(&[("instance_id", instance_id.as_str())])
                    .send()
                    .await
                {
                    Ok(response) if response.status().is_success() => {
                        match response.json::<PrefetchStatusResponse>().await {
                            Ok(status) if status.state.eq_ignore_ascii_case("READY") => {
                                tracing::debug!(%task_id, "LMCache prefetch became physically ready");
                                return;
                            }
                            Ok(status)
                                if status.state.eq_ignore_ascii_case("FAILED")
                                    || status.state.eq_ignore_ascii_case("CANCELLED")
                                    || status.state.eq_ignore_ascii_case("UNKNOWN") =>
                            {
                                tracing::debug!(%task_id, state = %status.state, "LMCache prefetch failed");
                                on_failure();
                                return;
                            }
                            Ok(_) => {}
                            Err(error) => {
                                tracing::debug!(%error, %task_id, "Invalid LMCache prefetch status response");
                            }
                        }
                    }
                    Ok(response) if response.status().is_client_error() => {
                        tracing::debug!(status = %response.status(), %task_id, "LMCache prefetch status rejected");
                        on_failure();
                        return;
                    }
                    Ok(response) if response.status().is_server_error() => {
                        tracing::debug!(status = %response.status(), %task_id, "LMCache prefetch task failed");
                        on_failure();
                        return;
                    }
                    Ok(response) => {
                        tracing::debug!(status = %response.status(), %task_id, "LMCache prefetch status unavailable");
                    }
                    Err(error) => {
                        // Do not turn a transient control-plane outage into a
                        // false physical failure. Keep the reservation until
                        // LMCache reports a terminal state.
                        tracing::debug!(%error, %task_id, "LMCache prefetch status polling failed");
                    }
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });
    }

    pub(crate) fn cancel(
        self: &Arc<Self>,
        request_id: String,
        worker: WorkerWithDpRank,
        route_epoch: u64,
    ) {
        let Some(endpoint) = self.config.endpoint.clone() else {
            return;
        };
        let Some(instance_id) = self.instance_id(worker) else {
            return;
        };
        let endpoint = if endpoint.contains("/prefetch_hint") {
            endpoint.replace("/prefetch_hint", "/cancel_prefetch_hint")
        } else if endpoint.ends_with("/v1/cxl/prefetch") {
            endpoint.replace("/v1/cxl/prefetch", "/cancel_prefetch_hint")
        } else {
            format!("{endpoint}/cancel_prefetch_hint")
        };
        let request = CancelRequest {
            instance_id,
            request_id,
            route_epoch,
        };
        let dispatcher = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(error) = dispatcher.client.post(endpoint).json(&request).send().await {
                tracing::debug!(%error, "LMCache prefetch cancellation failed");
            }
        });
    }
}
