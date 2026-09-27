# EGO_Mo: Pure-Inertial Camera Motion Estimation from Continuous IMU Streams

Code accompanying the manuscript *Pure-Inertial Camera Motion Estimation from
Continuous IMU Streams* (R. Li, N. Wei, Z. Lin, W. Liu, C. Fan; submitted to
IEEE Transactions on Instrumentation and Measurement).

A handheld stereo-endoscope rig carries a 200-Hz six-axis IMU. At inference the
estimator sees **only the continuous IMU stream and its timestamps** and outputs
3-D camera displacement increments and a relative translation trajectory.
Chessboard images are used only for calibration, training supervision and
evaluation.

## Repository layout

| Path | Content |
|---|---|
| `ego_capture/`, `capture_app.py`, `run_capture.bat` | Acquisition GUI: stereo cameras, USB and BLE IMUs, stepwise calibration and recording protocol |
| `ego_sim/`, `run_simulator.py` | Measurement digital twin: IMU simulation with the fitted sensor profile, OpenCV/Blender rendering |
| `tools/train_physnet.py` | PhysNet: anchor-frame pre-integration features, velocity-integration head, learnable lever arm, stillness gate, heteroscedastic head |
| `tools/train_external_architectures.py` | Protocol-matched adaptations of RoNIN-ResNet/LSTM, TLIO-ResNet and IMUNet |
| `tools/eval_session_cv.py` | Leave-one-session-out cross-validation, fold ensembles, blends, bootstrap CIs, pose-graph ATE |
| `tools/build_pose_gt.py`, `tools/export_gt_raw.py`, `tools/prepare_trajectory_data.py` | Chessboard PnP reference, label export, predeclared splits |
| `tools/reference_stage_validation.py`, `tools/sim_reference_validation.py` | Known-displacement validation of the reference (stage, caliper or simulated) |
| `tools/estimate_lever_arm.py`, `tools/register_lever_arm_measurement.py` | Physics-based and mechanical camera-IMU lever arm |
| `tools/final_prospective_evaluation.py`, `tools/check_prospective_acceptance.py` | Frozen-model manifest and one-shot prospective evaluation |
| `tools/twin_protocol_sweep.py`, `tools/stop_anchor_check.py`, `tools/twin_motion_class_eval.py`, `tools/twin_realistic_summary.py` | Realistic handheld twin, acquisition-protocol study (stop interval, label rate, ideal IMU), stop-anchored switching, error by motion class |
| `tools/make_paper_figures.py`, `tools/verify_manuscript.py`, `tools/recheck_metrics.py` | Figures, manuscript-to-artefact regression check, independent metric recomputation |
| `paper/` | LaTeX sources of the manuscript |
| `docs/` | Collection, simulation and submission protocols |

## Setup

```bash
pip install -r requirements.txt
pip install torch            # CUDA build recommended
```

Third-party reference implementations used for the adapted baselines are listed
in `external_sources.txt` (clone them into `external/`).

## Reproducing the paper

Recorded sessions are distributed separately (about 17 GB) and must be placed
in `datasets/`. Then:

```bash
python tools/prepare_trajectory_data.py          # predeclared split and labels
python tools/eval_session_cv.py --seeds 0 1 --cv-tag cvg --baseline-tag cvx \
       --output datasets/session_cv_gate_rerun.json   # do not overwrite the frozen file
python tools/make_paper_figures.py               # all figures in paper/figures
python tools/verify_manuscript.py                # every reported number vs artefacts
python tools/recheck_metrics.py                  # independent metric recomputation
```

See `REPRODUCIBILITY.md` for the full training commands and
`docs/submission_completion_protocol.md` for the prospective protocol.

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
