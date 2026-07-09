"""Shared pipeline logic.

Every per-combination script under PhaseA is a thin wrapper that sets
MODEL_KEY / DATASET_KEY and calls one function from this package - so the
real logic lives in exactly one place and is never copy-pasted.
"""
