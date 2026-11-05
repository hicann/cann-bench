# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""CPU reference exchange tests; also runnable directly without the NPU suite."""

from datetime import timedelta
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest

try:
    import torch
    import torch.distributed as dist
except ImportError:
    torch = None
    dist = None


SOURCE = (
    Path(__file__).resolve().parents[2] / "src/kernel_eval/eval/mc2_cpu_collectives.py"
)


def collectives_class():
    spec = importlib.util.spec_from_file_location(
        "mc2_cpu_collectives_under_test", SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CPUCollectives


def metadata(tensor, output_shape, send, recv, output_dtype=None):
    return {
        "input_shape": tuple(tensor.shape),
        "output_shape": tuple(output_shape),
        "input_dtype": tensor.dtype,
        "output_dtype": output_dtype or tensor.dtype,
        "send": send,
        "recv": recv,
    }


class BroadcastSimulation:
    """Feed each real source independently, preserving source order and shape."""

    ReduceOp = types.SimpleNamespace(SUM="sum")

    def __init__(self, sources, records, rank):
        self.sources, self.records, self.rank = sources, records, rank
        self.data_sources = []
        self.object_sources = []

    def broadcast_object_list(self, objects, src, group):
        assert group == "cpu-group"
        self.object_sources.append(src)
        if src == self.rank:
            self.records[src] = objects[0]
        objects[0] = self.records[src]

    def broadcast(self, tensor, src, group):
        assert group == "cpu-group"
        assert tensor.shape == self.sources[src].shape
        self.data_sources.append(src)
        tensor.copy_(self.sources[src])


@unittest.skipIf(torch is None, "PyTorch CPU is not installed")
class TestCPUAllToAll(unittest.TestCase):
    def exchange(self, sources, records, rank, output_shape, **kwargs):
        simulation = BroadcastSimulation(
            sources, [dict(record) for record in records], rank
        )
        cpu = collectives_class()(simulation, "cpu-group", rank, len(sources))
        output = torch.empty(output_shape, dtype=sources[rank].dtype)
        cpu.all_to_all_single(output, input=sources[rank], **kwargs)
        self.assertEqual(simulation.object_sources, list(range(len(sources))))
        return output, simulation

    def test_variable_splits_and_different_rank_shapes(self):
        sources = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            torch.tensor([[10.0, 20.0]]),
        ]
        snapshots = [source.clone() for source in sources]
        records = [
            metadata(sources[0], (1, 2), [1, 2], [1, 0]),
            metadata(sources[1], (3, 2), [0, 1], [2, 1]),
        ]
        for rank, expected in enumerate(
            ([[1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0], [10.0, 20.0]])
        ):
            with self.subTest(rank=rank):
                result, simulation = self.exchange(
                    sources,
                    records,
                    rank,
                    records[rank]["output_shape"],
                    input_split_sizes=records[rank]["send"],
                    output_split_sizes=records[rank]["recv"],
                )
                torch.testing.assert_close(result, torch.tensor(expected))
                self.assertEqual(simulation.data_sources, [0, 1])
        for source, snapshot in zip(sources, snapshots):
            torch.testing.assert_close(source, snapshot)

    def test_empty_sender_and_receiver(self):
        sources = [torch.empty(0, 2), torch.tensor([[10.0, 20.0], [30.0, 40.0]])]
        records = [
            metadata(sources[0], (2, 2), [0, 0], [0, 2]),
            metadata(sources[1], (0, 2), [2, 0], [0, 0]),
        ]
        for rank in range(2):
            result, simulation = self.exchange(
                sources,
                records,
                rank,
                records[rank]["output_shape"],
                input_split_sizes=records[rank]["send"],
                output_split_sizes=records[rank]["recv"],
            )
            torch.testing.assert_close(result, sources[1] if rank == 0 else sources[0])
            self.assertEqual(simulation.data_sources, [1])

    def test_legacy_leading_rank_slices_and_positional_input(self):
        sources = [
            torch.arange(8.0).reshape(2, 2, 2),
            torch.arange(10.0, 18.0).reshape(2, 2, 2),
        ]
        records = [metadata(source, source.shape, [1, 1], [1, 1]) for source in sources]
        simulation = BroadcastSimulation(sources, records, 1)
        cpu = collectives_class()(simulation, "cpu-group", 1, 2)
        output = torch.empty_like(sources[1])
        cpu.all_to_all_single(output, sources[1])
        torch.testing.assert_close(output, torch.stack([sources[0][1], sources[1][1]]))

    def test_equal_input_splits_with_unequal_rank_lengths(self):
        sources = [
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
        ]
        records = [
            metadata(sources[0], (3, 1), [1, 1], [1, 2]),
            metadata(sources[1], (3, 1), [2, 2], [1, 2]),
        ]
        result, _ = self.exchange(
            sources, records, 1, (3, 1), input_split_sizes=[], output_split_sizes=[1, 2]
        )
        torch.testing.assert_close(result, torch.tensor([[2.0], [30.0], [40.0]]))

    def test_invalid_local_counts_are_shared_before_data(self):
        source = torch.zeros(2, 1)
        for counts, message in (
            ([-1, 3], "nonnegative"),
            ([2], "expected 2 counts"),
            ([1, 2], "counts sum"),
            ([0.5, 1.5], "integers"),
            ([True, 1], "integers"),
        ):
            for direction in ("input_split_sizes", "output_split_sizes"):
                with self.subTest(counts=counts, direction=direction):
                    records = [
                        metadata(source, source.shape, [1, 1], [1, 1]) for _ in range(2)
                    ]
                    simulation = BroadcastSimulation([source, source], records, 0)
                    cpu = collectives_class()(simulation, "cpu-group", 0, 2)
                    with self.assertRaisesRegex(ValueError, message):
                        cpu.all_to_all_single(
                            torch.empty_like(source), source, **{direction: counts}
                        )
                    self.assertEqual(simulation.object_sources, [0, 1])
                    self.assertEqual(simulation.data_sources, [])

    def test_invalid_equal_split_and_scalar(self):
        for source in (torch.zeros(3, 1), torch.tensor(1.0)):
            other = torch.zeros(2, 1)
            simulation = BroadcastSimulation(
                [source, other], [{}, metadata(other, (2, 1), [1, 1], [1, 1])], 0
            )
            cpu = collectives_class()(simulation, "cpu-group", 0, 2)
            with self.assertRaises(ValueError):
                cpu.all_to_all_single(torch.empty(2, 1), source)
            self.assertEqual(simulation.data_sources, [])

    def test_remote_contract_errors_stop_every_rank_before_data(self):
        source = torch.zeros(2, 1)
        for changes, message in (
            ({"recv": [0, 2]}, "split mismatch"),
            ({"output_shape": (2, 2)}, "trailing shape mismatch"),
            ({"output_dtype": torch.float64}, "dtype mismatch"),
            ({"error": "invalid remote counts"}, "invalid remote counts"),
        ):
            with self.subTest(changes=changes):
                records = [
                    metadata(source, source.shape, [1, 1], [1, 1]) for _ in range(2)
                ]
                records[1].update(changes)
                simulation = BroadcastSimulation([source, source], records, 0)
                cpu = collectives_class()(simulation, "cpu-group", 0, 2)
                with self.assertRaisesRegex(ValueError, message):
                    cpu.all_to_all_single(torch.empty_like(source), source)
                self.assertEqual(simulation.data_sources, [])


def _gloo_worker(rank, init_method):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=30),
    )
    try:
        cpu = collectives_class()(dist, dist.group.WORLD, rank, 2)
        sources = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            torch.tensor([[10.0, 20.0]]),
        ]
        send, recv = ([1, 2], [0, 1]), ([1, 0], [2, 1])
        output = torch.empty(sum(recv[rank]), 2)
        cpu.all_to_all_single(
            output=output,
            input=sources[rank],
            output_split_sizes=recv[rank],
            input_split_sizes=send[rank],
        )
        expected = (
            torch.tensor([[1.0, 2.0]])
            if rank == 0
            else torch.tensor([[3.0, 4.0], [5.0, 6.0], [10.0, 20.0]])
        )
        torch.testing.assert_close(output, expected)

        source = (
            torch.empty(0, 2)
            if rank == 0
            else torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        )
        output = torch.empty(2 if rank == 0 else 0, 2)
        cpu.all_to_all_single(
            output,
            source,
            output_split_sizes=[0, 2] if rank == 0 else [0, 0],
            input_split_sizes=[0, 0] if rank == 0 else [2, 0],
        )
        torch.testing.assert_close(
            output,
            torch.tensor([[10.0, 20.0], [30.0, 40.0]])
            if rank == 0
            else torch.empty(0, 2),
        )

        # Only rank 0 has invalid counts. Both ranks must raise without leaving
        # the other in a data broadcast; a subsequent exchange must still work.
        try:
            cpu.all_to_all_single(
                torch.empty(2, 1),
                torch.ones(2, 1),
                input_split_sizes=[-1, 3] if rank == 0 else [1, 1],
            )
        except ValueError as exc:
            assert "nonnegative" in str(exc), str(exc)
        else:
            raise AssertionError("invalid rank 0 counts were accepted")

        source = torch.arange(rank * 10.0, rank * 10.0 + 8).reshape(2, 2, 2)
        output = torch.empty_like(source)
        cpu.all_to_all_single(output, source)
        expected = torch.stack(
            [
                torch.arange(8.0).reshape(2, 2, 2)[rank],
                torch.arange(10.0, 18.0).reshape(2, 2, 2)[rank],
            ]
        )
        torch.testing.assert_close(output, expected)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(
    dist is not None and dist.is_available() and dist.is_gloo_available(),
    "Gloo is not available",
)
class TestRealGlooAllToAll(unittest.TestCase):
    def test_two_ranks_variable_empty_legacy_and_shared_validation(self):
        with tempfile.TemporaryDirectory(prefix="mc2-gloo-") as directory:
            init_method = (Path(directory) / "rendezvous").as_uri()
            torch.multiprocessing.spawn(
                _gloo_worker, args=(init_method,), nprocs=2, join=True
            )


if __name__ == "__main__":
    unittest.main()
