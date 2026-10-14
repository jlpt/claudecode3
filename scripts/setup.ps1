# Fetches the pinned BanjoRecomp checkout with all of its submodules and applies the hybrid ray tracing patch to RT64.
# Safe to run more than once. Run from PowerShell: .\scripts\setup.ps1
$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Rt64Dir = Join-Path $Root "BanjoRecomp\lib\rt64"
$Patch = Join-Path $Root "patches\rt64-hybrid-raytracing.patch"

git -C $Root submodule update --init --recursive BanjoRecomp
if ($LASTEXITCODE -ne 0) { throw "Failed to fetch the BanjoRecomp submodules." }

git -C $Rt64Dir apply --reverse --check $Patch 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "The hybrid ray tracing patch is already applied to $Rt64Dir."
}
else {
    git -C $Rt64Dir apply --whitespace=nowarn $Patch
    if ($LASTEXITCODE -ne 0) { throw "Failed to apply $Patch." }
    Write-Host "Applied the hybrid ray tracing patch to $Rt64Dir."
}
