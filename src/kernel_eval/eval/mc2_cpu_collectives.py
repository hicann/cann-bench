# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""CPU FP32 reference collectives on a separate Gloo process group."""

import operator

import torch


def _split_sizes(sizes, length, world_size, name):
    sizes = [] if sizes is None else list(sizes)
    if not sizes:
        if length % world_size:
            raise ValueError(
                f"{name}: leading dimension {length} must divide world_size {world_size}"
            )
        return [length // world_size] * world_size
    if len(sizes) != world_size:
        raise ValueError(f"{name}: expected {world_size} counts, got {len(sizes)}")
    try:
        if any(isinstance(size, bool) for size in sizes):
            raise TypeError
        counts = [operator.index(size) for size in sizes]
    except TypeError as exc:
        raise ValueError(f"{name}: counts must be integers") from exc
    if any(size < 0 for size in counts):
        raise ValueError(f"{name}: counts must be nonnegative")
    if sum(counts) != length:
        raise ValueError(
            f"{name}: counts sum {sum(counts)} does not match leading dimension {length}"
        )
    return counts


class CPUCollectives:
    def __init__(self, dist, group, rank, world_size):
        self.dist, self.group = dist, group
        self.rank, self.world_size = rank, world_size
        self.ReduceOp = dist.ReduceOp

    def all_reduce(self, tensor, op=None):
        self.dist.all_reduce(tensor, op=op or self.ReduceOp.SUM, group=self.group)

    def _all_gather_base(self, output, tensor):
        parts = [torch.empty_like(tensor) for _ in range(self.world_size)]
        self.dist.all_gather(parts, tensor.contiguous(), group=self.group)
        output.copy_(torch.cat(parts, dim=0))

    def all_to_all_single(
        self, output, input, output_split_sizes=None, input_split_sizes=None
    ):
        """Exchange leading-dimension slices, including unequal or empty slices.

        Split counts have the same meaning as torch.distributed: input counts
        are indexed by destination and output counts by source. None or []
        selects equal splits. The original [world_size, ...] tensor layout is
        therefore the equal-split case with one leading row per peer.
        """
        # Share every rank's shapes/counts before moving data. This lets all
        # ranks reject an inconsistent contract together, rather than leaving
        # peers blocked in a broadcast when only one rank detects an error.
        try:
            input_shape, output_shape = tuple(input.shape), tuple(output.shape)
            if not input_shape or not output_shape:
                raise ValueError(
                    "all-to-all input and output must have a leading dimension"
                )
            local = {
                "input_shape": input_shape,
                "output_shape": output_shape,
                "input_dtype": input.dtype,
                "output_dtype": output.dtype,
                "send": _split_sizes(
                    input_split_sizes,
                    input_shape[0],
                    self.world_size,
                    "input_split_sizes",
                ),
                "recv": _split_sizes(
                    output_split_sizes,
                    output_shape[0],
                    self.world_size,
                    "output_split_sizes",
                ),
            }
        except (TypeError, ValueError) as exc:
            local = {"error": str(exc)}

        ranks = []
        for source in range(self.world_size):
            record = [local if source == self.rank else None]
            self.dist.broadcast_object_list(record, src=source, group=self.group)
            ranks.append(record[0])
        for rank, record in enumerate(ranks):
            if "error" in record:
                raise ValueError(f"CPU all-to-all rank {rank}: {record['error']}")
        for source, sender in enumerate(ranks):
            for target, receiver in enumerate(ranks):
                if sender["send"][target] != receiver["recv"][source]:
                    raise ValueError(
                        f"CPU all-to-all split mismatch from rank {source} to rank {target}"
                    )
                if sender["input_shape"][1:] != receiver["output_shape"][1:]:
                    raise ValueError(
                        f"CPU all-to-all trailing shape mismatch from rank {source} to rank {target}"
                    )
                if sender["input_dtype"] != receiver["output_dtype"]:
                    raise ValueError(
                        f"CPU all-to-all dtype mismatch from rank {source} to rank {target}"
                    )

        # Gloo need not implement all_to_all: broadcast one source at a time,
        # retaining at most one source tensor as temporary host storage.
        output_start = 0
        for source, sender in enumerate(ranks):
            payload = (
                input.contiguous().clone()
                if source == self.rank
                else torch.empty(
                    sender["input_shape"], dtype=sender["input_dtype"], device="cpu"
                )
            )
            if 0 not in sender["input_shape"]:
                self.dist.broadcast(payload, src=source, group=self.group)
            count = sender["send"][self.rank]
            input_start = sum(sender["send"][: self.rank])
            output[output_start : output_start + count].copy_(
                payload[input_start : input_start + count]
            )
            output_start += count
