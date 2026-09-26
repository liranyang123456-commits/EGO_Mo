$ErrorActionPreference = "Stop"
$root = "C:\Users\lry\EGO_Mo_fleet"
$python = "D:\anaconda\envs\track\python.exe"
$env:PYTHONPATH = $root
Set-Location $root
foreach ($context in @(1.0, 2.0)) {
    foreach ($seed in @(0, 1, 2)) {
        & $python -u tools\train_seq.py `
            --context $context `
            --epochs 80 `
            --seed $seed `
            --labels pose_gt_raw `
            --out traj_run_v10 `
            --split-file trajectory_split_20260924.json
        if ($LASTEXITCODE -ne 0) {
            throw "training failed: context=$context seed=$seed"
        }
    }
}
tar -czf C:\Users\lry\traj_run_v10_remote.tgz -C $root\datasets traj_run_v10
