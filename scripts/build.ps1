# Builds Banjo: Recompiled with the hybrid ray tracing patch on Windows.
#
# Run it from "Developer PowerShell for VS 2022" (Start menu) in the repository folder:
#   powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1 C:\path\to\your\banjo-kazooie.z64
#
# The ROM must be the NTSC-U 1.0 release dumped from a cartridge you own (z64, n64 or v64 byte order). It is only read to
# produce the decompressed copy that N64Recomp needs, which is written inside BanjoRecomp\ and is ignored by git.
#
# Requirements:
#   - Visual Studio 2022 with "Desktop development with C++", "C++ Clang Compiler for Windows" and "C++ CMake tools for Windows".
#   - Git for Windows, Python 3, make (choco install make) and LLVM (winget install LLVM.LLVM), which provides the
#     clang and ld.lld that compile the game patches, as in Banjo: Recompiled's own Windows builds.
param(
    [Parameter(Mandatory = $true)][string]$Rom,
    [ValidateSet("Release", "RelWithDebInfo", "Debug")][string]$BuildType = "Release"
)

# Native command failures are checked through $LASTEXITCODE. Windows PowerShell 5.1 turns anything git prints to
# stderr into an error record, so the error action can't be "Stop" here.
$ErrorActionPreference = "Continue"

# Versions the patch was built and tested against.
$N64RecompUrl = "https://github.com/N64Recomp/N64Recomp.git"
$N64RecompCommit = "2b6f05688de2abc7d86da5b4a89b84c2c6acbabe"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BkDir = Join-Path $Root "BanjoRecomp"
$ToolsDir = Join-Path $Root "build-tools"
$N64RecompDir = Join-Path $ToolsDir "N64Recomp"

function Fail([string]$Message) {
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Check([string]$What) {
    if ($LASTEXITCODE -ne 0) { Fail "$What failed (exit code $LASTEXITCODE)." }
}

function Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

Step "Checking tools"
if (-not (Get-Command cmake -ErrorAction SilentlyContinue) -or -not (Get-Command ninja -ErrorAction SilentlyContinue) -or -not $env:VCINSTALLDIR) {
    Fail "CMake, Ninja or the compiler environment wasn't found. Run this from 'Developer PowerShell for VS 2022' and make sure the 'C++ CMake tools for Windows' component is installed."
}

# The game itself is compiled with Visual Studio's clang-cl.
$VsLlvmBin = Join-Path $env:VCINSTALLDIR "Tools\Llvm\x64\bin"
$ClangCl = Join-Path $VsLlvmBin "clang-cl.exe"
if (-not (Test-Path $ClangCl)) {
    Fail "Visual Studio's Clang wasn't found in $VsLlvmBin. Install the 'C++ Clang Compiler for Windows' component in the Visual Studio Installer."
}

# The game patches are compiled for the N64 with clang and linked with ld.lld. Their makefile calls them by name, so the
# folder that has both goes first on the PATH.
$PatchesLlvmBin = $null
foreach ($Candidate in @((Join-Path $env:ProgramFiles "LLVM\bin"), $VsLlvmBin)) {
    if ((Test-Path (Join-Path $Candidate "clang.exe")) -and (Test-Path (Join-Path $Candidate "ld.lld.exe"))) {
        $PatchesLlvmBin = $Candidate
        break
    }
}

if (-not $PatchesLlvmBin) {
    Fail "clang and ld.lld weren't found. Install LLVM with 'winget install LLVM.LLVM' (or from https://github.com/llvm/llvm-project/releases), then open a new Developer PowerShell."
}

$env:PATH = "$PatchesLlvmBin;$env:PATH"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git wasn't found. Install Git for Windows from https://git-scm.com."
}

if (-not (Get-Command make -ErrorAction SilentlyContinue)) {
    Fail "make wasn't found. Install it with 'choco install make' (https://chocolatey.org), then open a new Developer PowerShell."
}

# The "python" command can be the Microsoft Store placeholder, so make sure it actually runs.
$Python = $null
foreach ($Candidate in @(@("py", "-3"), @("python"), @("python3"))) {
    if (Get-Command $Candidate[0] -ErrorAction SilentlyContinue) {
        $Arguments = @($Candidate | Select-Object -Skip 1)
        & $Candidate[0] @Arguments --version *> $null
        if ($LASTEXITCODE -eq 0) {
            $Python = $Candidate
            break
        }
    }
}

if (-not $Python) {
    Fail "Python 3 wasn't found. Install it from https://www.python.org (tick 'Add python.exe to PATH')."
}

if (-not (Test-Path -LiteralPath $Rom -PathType Leaf)) {
    Fail "ROM not found: $Rom"
}

$Rom = (Resolve-Path -LiteralPath $Rom).Path

Step "Fetching BanjoRecomp and applying the patch"
& (Join-Path $PSScriptRoot "setup.ps1")
Check "Setup"

Step "Decompressing the ROM"
$PythonArguments = @($Python | Select-Object -Skip 1) + @((Join-Path $PSScriptRoot "decompress_rom.py"), $Rom, (Join-Path $BkDir "banjo.us.v10.decompressed.z64"))
& $Python[0] @PythonArguments
Check "Decompressing the ROM"

Step "Building N64Recomp and RSPRecomp"
if (-not (Test-Path (Join-Path $N64RecompDir ".git"))) {
    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
    git clone $N64RecompUrl $N64RecompDir
    Check "Cloning N64Recomp"
}

git -C $N64RecompDir fetch --quiet origin
Check "Fetching N64Recomp"
git -C $N64RecompDir checkout --quiet $N64RecompCommit
Check "Checking out N64Recomp"
git -C $N64RecompDir submodule update --init --recursive
Check "Fetching the N64Recomp submodules"

cmake -S $N64RecompDir -B (Join-Path $N64RecompDir "cmake-build") -G Ninja -DCMAKE_BUILD_TYPE=Release
Check "Configuring N64Recomp"
cmake --build (Join-Path $N64RecompDir "cmake-build") --target N64RecompCLI RSPRecomp
Check "Building N64Recomp"
Copy-Item (Join-Path $N64RecompDir "cmake-build\N64Recomp.exe"), (Join-Path $N64RecompDir "cmake-build\RSPRecomp.exe") $BkDir -Force

Step "Recompiling the game code"
Push-Location $BkDir
try {
    & .\N64Recomp.exe banjo.us.rev0.toml
    Check "N64Recomp"
    & .\RSPRecomp.exe n_aspMain.us.rev0.toml
    Check "RSPRecomp"
}
finally {
    Pop-Location
}

Step "Building BanjoRecompiled ($BuildType)"
$BuildDir = Join-Path $BkDir "build-cmake"
# Same configuration as Banjo: Recompiled's Windows CI.
cmake -S $BkDir -B $BuildDir -G Ninja "-DCMAKE_BUILD_TYPE=$BuildType" "-DCMAKE_C_COMPILER=$ClangCl" "-DCMAKE_CXX_COMPILER=$ClangCl" `
    "-DCMAKE_CXX_FLAGS=-Xclang -fexceptions -Xclang -fcxx-exceptions"
Check "Configuring BanjoRecompiled"
cmake --build $BuildDir --target BanjoRecompiled
Check "Building BanjoRecompiled"

Write-Host ""
Write-Host "Done: $BuildDir\BanjoRecompiled.exe" -ForegroundColor Green
Write-Host "Start it from the BanjoRecomp folder so it finds the assets, and pick your normal ROM in the launcher:"
Write-Host "  cd `"$BkDir`""
Write-Host "  .\build-cmake\BanjoRecompiled.exe --show-console"
