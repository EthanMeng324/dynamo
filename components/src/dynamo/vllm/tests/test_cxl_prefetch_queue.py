# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the worker-side reactive CXL prefetch queue."""

import asyncio

from vllm.sampling_params import SamplingParams

from dynamo.vllm.handlers import BaseWorkerHandler, attach_cxl_prefetch_hint


class _TestHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        raise NotImplementedError


class _BlockingEngine:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.method = None
        self.args = None

    async def collective_rpc(self, method, args=()):
        self.method = method
        self.args = args
        self.started.set()
        await self.release.wait()
        return [{"scheduled": 1}]


def _handler(engine, queue_size=1):
    handler = object.__new__(_TestHandler)
    handler.engine_client = engine
    handler.shutdown_event = None
    handler._cxl_prefetch_queue = asyncio.Queue(maxsize=queue_size)
    handler._cxl_prefetch_worker_task = None
    handler._cxl_prefetch_rpc_timeout_s = 1.0
    handler.temp_dirs = []
    return handler


def test_endpoint_returns_before_collective_rpc_finishes():
    async def run_test():
        engine = _BlockingEngine()
        handler = _handler(engine)

        responses = [
            response
            async for response in handler.cxl_prefetch(
                {
                    "request_id": "request-1",
                    "token_ids": [1, 2, 3],
                    "dp_rank": 0,
                    "max_chunks": 1,
                    "candidate_block_indices": [2],
                }
            )
        ]

        assert responses[0]["status"] == "queued"
        assert not engine.started.is_set()

        await asyncio.wait_for(engine.started.wait(), timeout=1.0)
        assert engine.method == "cxl_prefetch"
        assert engine.args == ("request-1", [1, 2, 3], 0, 1, [2])
        handler.cleanup()
        await asyncio.sleep(0)

    asyncio.run(run_test())


def test_endpoint_drops_when_worker_queue_is_full():
    async def run_test():
        engine = _BlockingEngine()
        handler = _handler(engine, queue_size=1)
        handler._cxl_prefetch_queue.put_nowait(
            ("existing", [], 0, 1, [], "reactive", {})
        )

        responses = [
            response
            async for response in handler.cxl_prefetch(
                {
                    "request_id": "request-2",
                    "token_ids": [1, 2, 3],
                    "dp_rank": 0,
                    "max_chunks": 1,
                    "candidate_block_indices": [2],
                }
            )
        ]

        assert responses[0]["status"] == "dropped"
        assert responses[0]["reason"] == "queue_full"
        handler.cleanup()
        await asyncio.sleep(0)

    asyncio.run(run_test())


def test_route_hint_is_forwarded_through_kv_transfer_params():
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params.extra_args = {
        "kv_transfer_params": {"do_remote_decode": True},
        "existing_arg": "preserved",
    }
    request = {
        "routing": {
            "cxl_prefetch_hint": {
                "candidate_block_indices": [3, 1, 3],
                "max_chunks": 2,
                "route_decided_unix_ns": 123,
            }
        }
    }

    assert attach_cxl_prefetch_hint(request, sampling_params)
    assert sampling_params.extra_args == {
        "kv_transfer_params": {
            "do_remote_decode": True,
            "lmcache.cxl_prefetch_hint": {
                "candidate_block_indices": [1, 3],
                "max_chunks": 2,
                "route_decided_unix_ns": 123,
            },
        },
        "existing_arg": "preserved",
    }


def test_route_hint_candidate_bound_is_independent_of_batch_size():
    sampling_params = SamplingParams(max_tokens=1)
    request = {
        "routing": {
            "cxl_prefetch_hint": {
                "candidate_block_indices": list(range(12)),
                "max_chunks": 1,
            }
        }
    }

    assert attach_cxl_prefetch_hint(request, sampling_params)
    forwarded = sampling_params.extra_args["kv_transfer_params"][
        "lmcache.cxl_prefetch_hint"
    ]
    assert forwarded["candidate_block_indices"] == list(range(12))
    assert forwarded["max_chunks"] == 1
