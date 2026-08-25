# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the worker-side reactive CXL prefetch queue."""

import asyncio

from dynamo.vllm.handlers import BaseWorkerHandler


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
