$envs = Get-Content C:\Users\lry\.conda\environments.txt | Select-Object -Unique
foreach ($envPath in $envs) {
    $python = Join-Path $envPath "python.exe"
    if (-not (Test-Path $python)) {
        continue
    }
    Write-Output "PY=$python"
    & $python -c "import torch; print('TORCH='+torch.__version__); print('CUDA='+str(torch.cuda.is_available())); print('GPU='+torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>$null
}
