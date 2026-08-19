$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$envRoot = if ($env:FORMULA_OCR_CONDA_ENV) {
    $env:FORMULA_OCR_CONDA_ENV
} else {
    "C:\D\anaconda3\envs\formula_ocr"
}
$python = Join-Path $envRoot "python.exe"
$pyinstaller = Join-Path $envRoot "Scripts\pyinstaller.exe"
$specFile = Join-Path $root "FormulaOCR.spec"
$iconSvg = Join-Path $root "icon.svg"
$iconPng = Join-Path $root "icon.png"
$iconIco = Join-Path $root "icon.ico"

if (-not (Test-Path $python)) {
    throw "Python not found: $python"
}
if (-not (Test-Path $pyinstaller)) {
    & $python -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to install PyInstaller into $envRoot"
    }
}
if (-not (Test-Path $specFile)) {
    throw "PyInstaller spec not found: $specFile"
}

Set-Location $root
$runtimeBin = Join-Path $envRoot "Library\bin"
if (Test-Path $runtimeBin) {
    $env:PATH = $runtimeBin + ";" + $env:PATH
}

function Find-Browser {
    $candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft\Edge\Application\msedge.exe"),
        (Join-Path $env:ProgramFiles "Google\Chrome\Application\chrome.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }
    return $null
}

function Update-IconAssets {
    if (-not (Test-Path $iconSvg)) {
        return
    }

    $needsUpdate = $true
    if ((Test-Path $iconPng) -and (Test-Path $iconIco)) {
        $svgTime = (Get-Item $iconSvg).LastWriteTimeUtc
        $pngTime = (Get-Item $iconPng).LastWriteTimeUtc
        $icoTime = (Get-Item $iconIco).LastWriteTimeUtc
        $needsUpdate = ($pngTime -lt $svgTime) -or ($icoTime -lt $svgTime)
    }
    if (-not $needsUpdate) {
        return
    }

    $browser = Find-Browser
    if (-not $browser) {
        Write-Warning "Cannot render icon.svg because Edge/Chrome was not found."
        return
    }

    New-Item -ItemType Directory -Force -Path (Join-Path $root "build") | Out-Null
    $renderHtml = Join-Path $root "build\icon_render.html"
    $svg = Get-Content -LiteralPath $iconSvg -Raw -Encoding UTF8
    $html = @"
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html,body{margin:0;width:512px;height:512px;background:transparent;overflow:hidden}
svg{display:block;width:512px;height:512px}
</style>
</head>
<body>$svg</body>
</html>
"@
    Set-Content -LiteralPath $renderHtml -Value $html -Encoding UTF8
    $renderUri = [System.Uri]::new($renderHtml).AbsoluteUri
    & $browser `
        --headless `
        --disable-gpu `
        --hide-scrollbars `
        --default-background-color=00000000 `
        --window-size=512,512 `
        "--screenshot=$iconPng" `
        $renderUri | Out-Null

    if (-not (Test-Path $iconPng)) {
        throw "Browser did not produce icon.png"
    }
    & $python -c "from PIL import Image; img=Image.open(r'$iconPng').convert('RGBA'); img.save(r'$iconIco', sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)])"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to generate icon.ico"
    }
}

function Invoke-PackagedSelfTest {
    param(
        [string]$Name,
        [string]$Argument,
        [string]$Executable
    )
    Write-Host "Running packaged $Name..."
    & $Executable $Argument
    if ($LASTEXITCODE -ne 0) {
        throw "Packaged $Name failed with exit code $LASTEXITCODE"
    }
}

Update-IconAssets

Write-Host "Building FormulaOCR from: $root"
Write-Host "Conda environment: $envRoot"
Write-Host "Package mode: directory (onedir)"
& $pyinstaller --noconfirm --clean $specFile
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$builtExe = Join-Path $root "dist\FormulaOCR\FormulaOCR.exe"
if (-not (Test-Path $builtExe)) {
    throw "Built executable not found: $builtExe"
}
$internalDir = Join-Path $root "dist\FormulaOCR\_internal"
if (-not (Test-Path $internalDir)) {
    throw "PyInstaller output directory not found: $internalDir"
}

Invoke-PackagedSelfTest "runtime boundary self-test" "--runtime-self-test" $builtExe
Invoke-PackagedSelfTest "UI self-test" "--ui-self-test" $builtExe
Invoke-PackagedSelfTest "Word/MathML regression" "--word-mathml-self-test" $builtExe

$buildBytes = (
    Get-ChildItem -LiteralPath (Join-Path $root "dist\FormulaOCR") -Recurse -File |
        Measure-Object -Property Length -Sum
).Sum
$buildMiB = [Math]::Round($buildBytes / 1MB, 1)

Write-Host "Built: $builtExe"
Write-Host "Package size: $buildMiB MiB"
Write-Host "Models are downloaded on first use and cached under LocalAppData\FormulaOCR."
Write-Host "Distribute the complete dist\FormulaOCR directory; the EXE depends on _internal."
