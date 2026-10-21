#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Recovery ownership APIs and close ordering, without a device runtime."""

import threading
import unittest

from simpler.worker import Worker, _Lifecycle


class _NativeRecovery:
    def __init__(self):
        self.cancelled = threading.Event()
        self.waiting = threading.Event()
        self.closed = False
        self.block_wait = False
        self.calls = []

    def pending_recoveries(self):
        self.calls.append("pending")
        return ["episode"]

    def wait_for_recovery(self, timeout):
        self.calls.append(("wait", timeout))
        if self.block_wait:
            self.waiting.set()
            assert self.cancelled.wait(3)

    def recovery_graph(self, ticket):
        self.calls.append(("graph", ticket))
        return "graph"

    def give_up_recovery(self, ticket, reason):
        self.calls.append(("give_up", ticket, reason))
        return True

    def cancel_recoveries(self):
        self.cancelled.set()

    def close(self):
        assert self.cancelled.is_set()
        self.closed = True


class RetainedRecoveryFacadeTest(unittest.TestCase):
    def owner(self, *, automatic=False):
        worker = Worker(
            level=3,
            platform="a5",
            device_ids=[0],
            num_sub_workers=0,
            enable_operator_recovery=True,
            automatic_operator_recovery=automatic,
        )
        native = _NativeRecovery()
        worker._worker = native
        worker._effective_operator_recovery = True
        worker._lifecycle = _Lifecycle.READY
        worker._init_owner_thread = threading.current_thread()
        return worker, native

    def test_owner_apis_delegate_under_the_operation_lease(self):
        worker, native = self.owner()
        try:
            self.assertEqual(worker.pending_recoveries(), ["episode"])
            self.assertIsNone(worker.wait_for_recovery(0.25))
            self.assertEqual(worker.recovery_graph("ticket"), "graph")
            self.assertTrue(worker.give_up_recovery("ticket", "owner rejected"))
            self.assertEqual(
                native.calls, ["pending", ("wait", 0.25), ("graph", "ticket"), ("give_up", "ticket", "owner rejected")]
            )
            self.assertEqual(worker._active_ops, 0)
        finally:
            worker.close()

    def test_disabled_owner_is_refused(self):
        worker, _native = self.owner()
        try:
            worker._effective_operator_recovery = False
            with self.assertRaisesRegex(RuntimeError, "enabled A5 local L3"):
                worker.pending_recoveries()
        finally:
            worker._effective_operator_recovery = True
            worker.close()

    def test_automatic_owner_refuses_manual_give_up(self):
        worker, native = self.owner(automatic=True)
        try:
            with self.assertRaisesRegex(RuntimeError, "automatic RecoveryCoordinator"):
                worker.give_up_recovery("ticket", "manual override")
            self.assertEqual(native.calls, [])
            self.assertEqual(worker._active_ops, 0)
        finally:
            worker.close()

    def test_close_cancels_before_waiting_for_a_leased_recovery_owner(self):
        worker, native = self.owner()
        native.block_wait = True
        results = []
        thread = threading.Thread(target=lambda: results.append(worker.wait_for_recovery(3)))
        thread.start()
        try:
            self.assertTrue(native.waiting.wait(3))
            worker.close()
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(results, [None])
            self.assertTrue(native.closed)
        finally:
            native.cancelled.set()
            thread.join(3)
            worker.close()

    def test_close_cancels_before_draining_accepted_run_fences(self):
        worker, native = self.owner()

        class Fence:
            def wait(self, _timeout):
                assert native.cancelled.is_set()
                worker._accepted_run_handles.discard(self)

        worker._accepted_run_handles.add(Fence())
        worker.close()
        self.assertTrue(native.closed)


if __name__ == "__main__":
    unittest.main()
