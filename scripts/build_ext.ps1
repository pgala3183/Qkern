#Requires -Version 5.1
<#
.SYNOPSIS
  Build the QKern CUDA Python extension on Windows (VS Build Tools + CUDA).
#>
$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
if (-not (Test-Path -LiteralPath "$CudaRoot\include\cuda_runtime.h")) {
  throw "CUDA toolkit not found at $CudaRoot"
}
$env:CUDA_PATH = $CudaRoot
$env:CUDA_HOME = $CudaRoot
$env:CUDAToolkit_ROOT = $CudaRoot
$env:CUDA_PATH_V13_3 = $CudaRoot
$env:PATH = "$CudaRoot\bin;" + $env:PATH

$VsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $VsDevCmd)) {
  throw "VsDevCmd.bat not found"
}

$Bat = @"
@echo off
call "$VsDevCmd" -arch=x64 >nul
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=7.5
cd /d "$RepoRoot"
python -m pip install wheel setuptools -q
python setup.py build_ext --inplace
if errorlevel 1 exit /b 1
python -m pip install -e ".[dev]" --no-build-isolation -q
if errorlevel 1 exit /b 1
python -c "from qkern import fp16_gemv, is_cuda_extension_available; import torch; print('ext', is_cuda_extension_available()); print('cuda', torch.cuda.is_available())"
"@
$TempBat = Join-Path $env:TEMP "qkern_build_ext.bat"
Set-Content -Path $TempBat -Value $Bat -Encoding ASCII
& cmd.exe /c $TempBat
if ($LASTEXITCODE -ne 0) { throw "Extension build failed" }
Write-Host "Extension build OK"
