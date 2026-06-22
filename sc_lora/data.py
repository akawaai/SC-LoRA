from __future__ import annotations

from typing import Any


def split_batch(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, dict):
        inputs = batch.get("inputs", batch)
        targets = batch.get("targets")
        if "labels" in batch and targets is None:
            targets = batch["labels"]
        return inputs, targets
    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            return batch[0], batch[1]
        if len(batch) == 1:
            return batch[0], None
    return batch, None


def batch_size_of_inputs(inputs: Any) -> int:
    if isinstance(inputs, dict):
        for value in inputs.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                return int(value.shape[0])
        raise ValueError("Could not infer batch size from dict inputs.")
    if hasattr(inputs, "shape") and len(inputs.shape) > 0:
        return int(inputs.shape[0])
    raise ValueError("Could not infer batch size from inputs.")
