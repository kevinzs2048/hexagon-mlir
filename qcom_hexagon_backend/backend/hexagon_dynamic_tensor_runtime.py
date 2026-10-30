# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Runtime-only owned tensor decoding for the bounded buffer ABI.

No model math or device allocation. Each returned tensor owns a fresh writable
bytearray through PyTorch's frombuffer owner reference; neither the response
nor a reusable shared-memory mapping can overwrite a previous result.
"""
import sys

from .hexagon_dynamic_buffer_io import layout_for


def unpack_owned_torch_outputs(contract, values, response):
    """Copy each output once into independently owned CPU-visible storage.

The established decoder slices bytes, copies them into bytearrays, then clones
the tensors. This variant avoids the slice/clone copies, NOT the required
ownership copy. It must not expose views of a reusable transport buffer.
"""
    import torch

    if sys.byteorder != 'little':
        raise ValueError('The buffer ABI requires a little-endian host')
    if not isinstance(response, (bytes, bytearray, memoryview)):
        raise ValueError('A contiguous response byte buffer is required')
    layout = layout_for(contract, values)
    view = memoryview(response)
    try:
        if not view.c_contiguous:
            raise ValueError('A contiguous response byte buffer is required')
        raw = view.cast('B')
        try:
            if len(raw) != layout['output_bytes']:
                raise ValueError('Actual response length differs from compiled layout')
            dtypes = {'float32': torch.float32, 'int64': torch.int64}
            outputs = []
            for row in layout['outputs']:
                owned = bytearray(raw[row['offset']:row['offset']+row['bytes']])
                outputs.append(torch.frombuffer(owned, dtype=dtypes[row['dtype']]).reshape(row['shape']))
            return tuple(outputs)
        finally:
            raw.release()
    finally:
        view.release()
