# Fetches the pinned BanjoRecomp checkout with all of its submodules and applies the hybrid ray tracing patch to RT64.
# Safe to run more than once. Run from PowerShell in the repository folder:
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
# Native command failures are checked through $LASTEXITCODE. Windows PowerShell 5.1 turns anything git prints to
# stderr into an error record, so the error action can't be "Stop" here.
$ErrorActionPreference = "Continue"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Rt64Dir = Join-Path $Root "BanjoRecomp\lib\rt64"
$Patch = Join-Path $Root "patches\rt64-hybrid-raytracing.patch"

# Check the sources out exactly as they are stored (LF line endings) so the patch applies cleanly, regardless of
# Git for Windows' core.autocrlf setting.
git -c core.autocrlf=false -C $Root submodule update --init --recursive BanjoRecomp
if ($LASTEXITCODE -ne 0) { throw "Failed to fetch the BanjoRecomp submodules." }

# --ignore-whitespace covers checkouts that were already converted to CRLF line endings.
git -C $Rt64Dir apply --reverse --check --ignore-whitespace $Patch 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "The hybrid ray tracing patch is already applied to $Rt64Dir."
    exit 0
}

git -C $Rt64Dir apply --whitespace=nowarn $Patch 2>$null
if ($LASTEXITCODE -ne 0) {
    git -C $Rt64Dir apply --whitespace=nowarn --ignore-whitespace $Patch
    if ($LASTEXITCODE -ne 0) { throw "Failed to apply $Patch to $Rt64Dir." }
}

Write-Host "Applied the hybrid ray tracing patch to $Rt64Dir."
