"""Geometric extrinsic residual, and the camera-pose corrector.

The pose model is inertial.MotionTrajectoryCorrector. The dual-stream residual
mapper is the older extrinsic experiment and is not the ego-motion estimator.
"""

from .decouple import DecoupledEgomotion, decomposition_loss
from .inertial import MotionTrajectoryCorrector, correction_loss, initialize_from_static, snap_to_pnp
from .models import DualStreamFrameMapper, FrameMapMLP, FrameMappingSystem, build_mapper
from .prior import apply_residual, compose_camera_from_imu
from .so3 import geodesic_loss, rot6d_to_R

__all__ = [
    "DualStreamFrameMapper",
    "FrameMapMLP",
    "FrameMappingSystem",
    "DecoupledEgomotion",
    "MotionTrajectoryCorrector",
    "decomposition_loss",
    "apply_residual",
    "build_mapper",
    "compose_camera_from_imu",
    "correction_loss",
    "geodesic_loss",
    "initialize_from_static",
    "rot6d_to_R",
    "snap_to_pnp",
]
