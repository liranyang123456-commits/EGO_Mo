"""Synthetic stereo-camera and IMU dataset generator."""

from .core import SimConfig, SyntheticSequence, build_sequence, export_sequence

__all__ = ["SimConfig", "SyntheticSequence", "build_sequence", "export_sequence"]
