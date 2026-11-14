[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$VpsHost,

    [string]$RemoteDirectory = "/opt/bp/training-output/latest-candidate",

    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,

    [switch]$Approve
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $Approve) {
    throw "Review the candidate in MLflow, then rerun with -Approve to promote it locally."
}

$modelDirectory = Join-Path $ProjectRoot "models"
$stagingDirectory = Join-Path $ProjectRoot (".model-download\" + [guid]::NewGuid().ToString("N"))
$backupDirectory = Join-Path $modelDirectory ("backups\" + (Get-Date -Format "yyyyMMdd-HHmmss"))
$requiredFiles = @("btts_model.joblib", "btts_feature_columns.json")
$optionalFiles = @("training_summary.json", "model_comparison.csv", "threshold_evaluation.csv")

New-Item -ItemType Directory -Force -Path $stagingDirectory | Out-Null
try {
    foreach ($file in $requiredFiles) {
        $remotePath = "${VpsHost}:$RemoteDirectory/$file"
        & scp $remotePath (Join-Path $stagingDirectory $file)
        if ($LASTEXITCODE -ne 0) {
            throw "scp failed while downloading $file from $VpsHost."
        }
    }

    foreach ($file in $optionalFiles) {
        $remotePath = "${VpsHost}:$RemoteDirectory/$file"
        & scp -q $remotePath (Join-Path $stagingDirectory $file)
    }

    $incomingModel = Join-Path $stagingDirectory "btts_model.joblib"
    $incomingColumns = Join-Path $stagingDirectory "btts_feature_columns.json"
    if ((Get-Item -LiteralPath $incomingModel).Length -le 0) {
        throw "Downloaded model file is empty."
    }
    Get-Content -Raw -LiteralPath $incomingColumns | ConvertFrom-Json | Out-Null

    New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null
    $existingFiles = $requiredFiles | Where-Object {
        Test-Path -LiteralPath (Join-Path $modelDirectory $_)
    }
    if ($existingFiles) {
        New-Item -ItemType Directory -Force -Path $backupDirectory | Out-Null
        foreach ($file in $existingFiles) {
            Copy-Item -LiteralPath (Join-Path $modelDirectory $file) -Destination $backupDirectory
        }
        Write-Host "Backed up the previous local model to $backupDirectory"
    }

    foreach ($file in $requiredFiles) {
        Copy-Item -Force -LiteralPath (Join-Path $stagingDirectory $file) -Destination $modelDirectory
    }

    Write-Host "Promoted the approved VPS candidate into $modelDirectory"
}
finally {
    if (Test-Path -LiteralPath $stagingDirectory) {
        Remove-Item -Recurse -Force -LiteralPath $stagingDirectory
    }
}
