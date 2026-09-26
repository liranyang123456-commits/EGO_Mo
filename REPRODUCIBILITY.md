# Reproducing EGO_Mo

## Scope

The learned deployment input is the continuous six-axis IMU stream and its
timestamps. Chessboard images are used only for calibration, training
supervision, and evaluation. The present network estimates 3-D displacement;
relative orientation is obtained by gyroscope integration.

## Environment

```powershell
conda create -n ego-mo python=3.13
conda activate ego-mo
pip install -r requirements.txt
$env:PYTHONPATH="E:\EGO_Mo"
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

Record exact package versions for an archive with:

```powershell
python -m pip freeze > release_requirements_lock.txt
```

## Audit and split

```powershell
python tools/audit_sessions.py
python tools/build_pose_gt.py
python tools/prepare_trajectory_data.py
python tools/verify_manuscript.py
```

The frozen prospective protocol is
`datasets/study_protocol_v2.lock.json`. New theme-T recordings enter the
prospective test only if their `session_meta.json` carries the matching
`protocol_id`.

## Core training and protocol-matched baseline

```powershell
python tools/train_physnet.py train --tag cv_<heldout> --seed 0 `
  --out physnet_cv --weight-decay 0.1 --still-weight 1 --gate `
  --holdout <heldout>

python tools/train_external_architectures.py --arch imunet `
  --only real_only --seed 0 --holdout <heldout> `
  --stem imunet_cv_<heldout>_s0
```

Repeat for every non-test recording and seeds 0 and 1. The final evaluator is
`tools/final_prospective_evaluation.py`; it refuses to run until the required
prospective sessions are present and the model bundle matches the frozen
protocol.

## Independent geometry checks

```powershell
python tools/reference_stage_validation.py init
python tools/reference_stage_validation.py analyze

python tools/calibrate_stereo_extrinsic.py --session <stereo_ext_session>

python tools/register_lever_arm_measurement.py init
python tools/register_lever_arm_measurement.py analyze
```

No geometric result is accepted by these tools when the plan is incomplete or
its held-out acceptance criteria fail.

## Paper

```powershell
python tools/make_paper_figures.py
python tools/verify_manuscript.py
cd paper
pdflatex -jobname=main_submission main.tex
bibtex main_submission
pdflatex -jobname=main_submission main.tex
pdflatex -jobname=main_submission main.tex
```

## Release

```powershell
python tools/build_release_bundle.py
```

Upload the generated ZIP to a public repository/Zenodo, then replace
`TO_BE_ASSIGNED` in `CITATION.cff` and the manuscript data-availability
statement with the immutable repository URL and DOI.
