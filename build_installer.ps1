param(
    [switch]$SkipAppBuild
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$version = (Get-Content -LiteralPath (Join-Path $root "VERSION") -Raw).Trim()
$installerScript = Join-Path $root "installer\FormulaOCR.iss"
$chineseMessagesUrl = (
    "https://raw.githubusercontent.com/jrsoftware/issrc/" +
    "69a2554fc9551f1d3da8df8ba659007dea3f906f/" +
    "Files/Languages/ChineseSimplified.isl"
)
$chineseMessagesSha256 = "e0b0b350e2245f3c5e65586dfe43d574f6e7f06f2261149aba284954b3fc9a8d"

if (-not $SkipAppBuild) {
    & (Join-Path $root "build_exe.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "Application build failed with exit code $LASTEXITCODE"
    }
}

$builtExe = Join-Path $root "dist\FormulaOCR\FormulaOCR.exe"
if (-not (Test-Path $builtExe)) {
    throw "Application directory build was not found: $builtExe"
}

$isccCandidates = @(
    $env:FORMULA_OCR_ISCC,
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
$iscc = $isccCandidates | Select-Object -First 1
if (-not $iscc) {
    throw "Inno Setup 6 was not found. Install JRSoftware.InnoSetup first."
}

$env:FORMULA_OCR_REPO_ROOT = $root
New-Item -ItemType Directory -Force -Path (Join-Path $root "dist\installer") | Out-Null
$translationRoot = Join-Path ([IO.Path]::GetTempPath()) (
    "FormulaOCR-Inno-" + [Guid]::NewGuid().ToString("N")
)
$translationFile = Join-Path $translationRoot "ChineseSimplified.isl"
try {
    New-Item -ItemType Directory -Force -Path $translationRoot | Out-Null
    & curl.exe --location --fail --silent --show-error `
        --retry 3 --retry-delay 2 `
        --output $translationFile $chineseMessagesUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to download the Inno Setup translation: exit code $LASTEXITCODE"
    }
    $translationHash = (
        Get-FileHash -LiteralPath $translationFile -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($translationHash -ne $chineseMessagesSha256) {
        throw "Inno Setup Chinese translation SHA-256 mismatch: $translationHash"
    }
    $env:FORMULA_OCR_CHINESE_ISL = $translationFile
    & $iscc "/Q" $installerScript
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:FORMULA_OCR_CHINESE_ISL -ErrorAction SilentlyContinue
    if (Test-Path $translationRoot) {
        Remove-Item -LiteralPath $translationRoot -Recurse -Force
    }
}

$installer = Join-Path $root "dist\installer\FormulaOCRSetup-$version.exe"
if (-not (Test-Path $installer)) {
    throw "Installer was not produced: $installer"
}
$hash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant()
$checksum = "$installer.sha256"
[IO.File]::WriteAllText(
    $checksum,
    "$hash  $([IO.Path]::GetFileName($installer))`n",
    [Text.UTF8Encoding]::new($false)
)

Write-Host "Built installer: $installer"
Write-Host "SHA-256: $hash"
