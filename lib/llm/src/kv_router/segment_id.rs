// SPDX-License-Identifier: Apache-2.0
//! Logical KV segment identity shared by routing and LMCache adapters.
//!
//! Only namespace fields and the deterministic token/block hash are carried;
//! no worker-local pointer, CXL offset, or request id is part of the identity.

use serde::{Deserialize, Serialize};

use super::protocols::{LocalBlockHash, WorkerId};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SegmentNamespace {
    pub model_hash: u64,
    #[serde(default)]
    pub model_revision: u32,
    #[serde(default)]
    pub tensor_parallel_rank: WorkerId,
    #[serde(default)]
    pub pipeline_parallel_rank: u16,
    #[serde(default)]
    pub kv_layout_version: u16,
    #[serde(default)]
    pub kv_dtype: String,
    #[serde(default)]
    pub chunk_size: u32,
    #[serde(default)]
    pub layer_group: u16,
}

impl SegmentNamespace {
    pub fn canonical(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}:{}:{}:{}",
            self.model_hash,
            self.model_revision,
            self.tensor_parallel_rank,
            self.pipeline_parallel_rank,
            self.kv_layout_version,
            self.kv_dtype,
            self.chunk_size,
            self.layer_group
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SegmentId {
    pub namespace: SegmentNamespace,
    /// Dynamo currently uses a u64 block hash.  LMCache adapters can carry a
    /// wider hash in their opaque payload and map it to this value explicitly.
    pub token_chunk_hash: u64,
}

impl SegmentId {
    pub fn from_block_hash(namespace: SegmentNamespace, hash: LocalBlockHash) -> Self {
        Self {
            namespace,
            token_chunk_hash: hash.0,
        }
    }

    pub fn canonical(&self) -> String {
        format!(
            "{}:{:016x}",
            self.namespace.canonical(),
            self.token_chunk_hash
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SegmentBlockMapping {
    pub segment_id: SegmentId,
    pub router_block_hashes: Vec<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_identity_excludes_request_and_address_state() {
        let ns = SegmentNamespace {
            model_hash: 7,
            model_revision: 1,
            tensor_parallel_rank: 2,
            pipeline_parallel_rank: 0,
            kv_layout_version: 1,
            kv_dtype: "bf16".to_string(),
            chunk_size: 256,
            layer_group: 0,
        };
        let id = SegmentId::from_block_hash(ns, LocalBlockHash(0xab));
        assert_eq!(id.canonical(), "7:1:2:0:1:bf16:256:0:00000000000000ab");
    }
}
