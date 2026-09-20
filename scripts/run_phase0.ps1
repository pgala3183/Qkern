#Requires -Version 5.1
<#
.SYNOPSIS
  Phase 0: configure, build, and run CUDA / NVRTC smoke checks on Windows.
#>
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$CudaRootCandidates = @(
  $(if ($env:CUDAToolkit_ROOT -and (Test-Path -LiteralPath $env:CUDAToolkit_ROOT)) { $env:CUDAToolkit_ROOT }),
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

# Force array so a single match is not unwrapped to a string ( [0] would become 'C' ).
$CudaRootCandidates = @($CudaRootCandidates)
if ($CudaRootCandidates.Count -lt 1) {
  throw "CUDA Toolkit root not found. Set CUDAToolkit_ROOT to the toolkit directory (not ...\bin)."
}
$CudaRoot = $CudaRootCandidates[0]
$env:CUDAToolkit_ROOT = $CudaRoot

# This machine (and some CUDA installs) mis-set CUDA_PATH / CUDA_PATH_V* to bin or libnvvp.
# VS CUDA .targets reads those variables; force the real toolkit root for this process.
$env:CUDA_PATH = $CudaRoot
$env:CUDA_PATH_V13_3 = $CudaRoot
$env:CUDA_PATH_V12_8 = $CudaRoot
$env:CudaToolkitDir = $CudaRoot
$env:PATH = "$CudaRoot\bin;" + $env:PATH

$CMake = if (Test-Path "C:\Program Files\CMake\bin\cmake.exe") {
  "C:\Program Files\CMake\bin\cmake.exe"
} elseif (Get-Command cmake -ErrorAction SilentlyContinue) {
  (Get-Command cmake).Source
} else {
  throw "cmake not found. Install Kitware CMake and re-run."
}

$VsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $VsDevCmd)) {
  throw "VsDevCmd.bat not found at $VsDevCmd. Install VS 2022 Build Tools with C++ workload."
}

Write-Host "Repo:        $RepoRoot"
Write-Host "CUDAToolkit: $CudaRoot"
Write-Host "CMake:       $CMake"

$ConfigureCmd = @(
  "`"$CMake`"",
  "-S `"$RepoRoot`"",
  "-B `"$RepoRoot\build`"",
  "-G `"Visual Studio 17 2022`"",
  "-A x64",
  "-DCUDAToolkit_ROOT=`"$CudaRoot`"",
  "-DCMAKE_CUDA_ARCHITECTURES=75"
) -join " "

$BuildCmd = "`"$CMake`" --build `"$RepoRoot\build`" --config Release"

$Bat = @"
@echo off
call "$VsDevCmd" -arch=x64 >nul
$ConfigureCmd
if errorlevel 1 exit /b 1
$BuildCmd
if errorlevel 1 exit /b 1
"@

$TempBat = Join-Path $env:TEMP "qkern_phase0_build.bat"
Set-Content -Path $TempBat -Value $Bat -Encoding ASCII
Write-Host "Configuring and building..."
& cmd.exe /c $TempBat
if ($LASTEXITCODE -ne 0) {
  throw "Phase 0 CMake build failed with exit code $LASTEXITCODE"
}

$DeviceQuery = Join-Path $RepoRoot "build\Release\qkern_device_query.exe"
$NvrtcSmoke = Join-Path $RepoRoot "build\Release\qkern_nvrtc_smoke.exe"

Write-Host "`n=== qkern_device_query ==="
& $DeviceQuery
if ($LASTEXITCODE -ne 0) { throw "device_query failed" }

Write-Host "`n=== qkern_nvrtc_smoke ==="
& $NvrtcSmoke
if ($LASTEXITCODE -ne 0) { throw "nvrtc_smoke failed" }

Write-Host "`n=== python diagnostics ==="
python -m pip install -e "$RepoRoot" -q
python -m qkern.diagnostics

Write-Host "`nPhase 0 smoke checks completed."
