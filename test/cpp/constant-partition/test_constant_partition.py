# SPDX-License-Identifier: BSD-3-Clause
"""Execute the constant-partition pass without allocating large tensor payloads."""
import json
import os
import random
import subprocess

import pytest

LIMIT = 500_000_000


def run_probe(*args):
    binary = os.environ.get("HEXAGON_CONSTANT_PARTITION_PROBE")
    if not binary:
        pytest.fail("Set HEXAGON_CONSTANT_PARTITION_PROBE to the built C++ probe")
    result = subprocess.run(
        [binary, *map(str, args)], capture_output=True, text=True,
        timeout=20, check=True,
    )
    return json.loads(result.stdout)


def probe(sizes):
    record = run_probe(*sizes)
    assert all(m["module_attribute_preserved"] for m in record["modules"])
    return [m["sizes"] for m in record["modules"]]


def expected(sizes):
    groups = []
    for size in sizes:
        if not groups or sum(groups[-1]) + size > LIMIT:
            groups.append([])
        groups[-1].append(size)
    return groups


@pytest.mark.parametrize("sizes", [
    [], [1], [LIMIT], [LIMIT, 1], [LIMIT - 1, 1],
    [400_000_000, 200_000_000, 300_000_000, 1],
    [100, 622_329_856, 443_000_000, 155_000_000],
    [622_329_856, 1], [1 << 31, 1], [1 << 32, LIMIT, 1], [1 << 40, 1],
])
def test_partition_boundaries(sizes):
    groups = probe(sizes)
    assert groups == expected(sizes)
    assert [value for group in groups for value in group] == sizes
    assert all(sum(group) <= LIMIT or len(group) == 1 for group in groups)


@pytest.mark.parametrize("seed", range(10))
def test_mixed_sizes(seed):
    rng = random.Random(seed)
    choices = [1, 50_000_000, 400_000_000, LIMIT - 1, LIMIT, 622_329_856, 1 << 31]
    sizes = [rng.choice(choices) for _ in range(20)]
    assert probe(sizes) == expected(sizes)


def test_resources_survive_all_partition_round_trips():
    assert run_probe("--resource") == {
        "resource_round_trip": True,
        "resource_count": 2,
        "partition_count": 3,
    }
