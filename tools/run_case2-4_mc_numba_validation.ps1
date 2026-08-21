[CmdletBinding()]
param(
    [string]$OutputRoot = "",
    [ValidateSet("auto", "batch", "interactive")]
    [string]$ParallelPolicy = "batch",
    [string]$Workers = "auto",
    [switch]$DryRun,
    [switch]$SkipCase2,
    [switch]$SkipCase4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$case2Package = Join-Path $projectRoot "dist\sustainability_2024_case2_strict_fos_005_20260725"
$case4Package = Join-Path $projectRoot "dist\sustainability_2024_case1-4_auto_srm_speed_guarded_20260612"
$case2Runner = Join-Path $case2Package "run_case2_strict_fos_005.ps1"
$case4Runner = Join-Path $case4Package "run_case1-4_srm.ps1"

foreach ($requiredPath in @($case2Runner, $case4Runner)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required validation runner was not found: $requiredPath"
    }
}

if ($OutputRoot) {
    $runDirectory = if ([IO.Path]::IsPathRooted($OutputRoot)) {
        [IO.Path]::GetFullPath($OutputRoot)
    } else {
        [IO.Path]::GetFullPath((Join-Path $projectRoot $OutputRoot))
    }
} else {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $runDirectory = Join-Path $projectRoot "runs\case2-4_mc_numba_validation_$stamp"
}
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$runDirectory = [IO.Path]::GetFullPath($runDirectory)
$runDirectory | Set-Content -LiteralPath (Join-Path $projectRoot "last_case2-4_mc_numba_validation.txt") -Encoding ascii

function Get-PropertyValue {
    param(
        [object]$Object,
        [string]$Name,
        [object]$Default = $null
    )
    if ($null -eq $Object) {
        return $Default
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $Default
    }
    return $property.Value
}

function Get-SrmMetrics {
    param([string]$SummaryPath)

    $metrics = [ordered]@{
        factor_of_safety = $null
        stable_factor = $null
        failed_factor = $null
        trial_count = 0
        mc_numba_to_python_fallback_count = 0
        mc_numba_regularized_projection_count = 0
        mc_regularized_projection_count = 0
    }
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        return [pscustomobject]$metrics
    }
    $summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    $stage = @((Get-PropertyValue $summary "stages" @())) | Select-Object -First 1
    $solver = Get-PropertyValue $stage "solver"
    $srm = Get-PropertyValue $solver "srm"
    $trials = @((Get-PropertyValue $srm "trials" @()))
    $metrics.factor_of_safety = Get-PropertyValue $srm "factor_of_safety"
    $metrics.stable_factor = Get-PropertyValue $srm "bracket_stable_factor" (Get-PropertyValue $srm "stable_factor")
    $metrics.failed_factor = Get-PropertyValue $srm "bracket_failed_factor" (Get-PropertyValue $srm "failed_factor")
    $metrics.trial_count = $trials.Count
    foreach ($trial in $trials) {
        $metrics.mc_numba_to_python_fallback_count += [int](Get-PropertyValue $trial "mc_numba_to_python_fallback_count" 0)
        $metrics.mc_numba_regularized_projection_count += [int](Get-PropertyValue $trial "mc_numba_regularized_projection_count" 0)
        $metrics.mc_regularized_projection_count += [int](Get-PropertyValue $trial "mc_regularized_projection_count" 0)
    }
    return [pscustomobject]$metrics
}

function Invoke-NestedRunner {
    param(
        [string]$ScriptPath,
        [object[]]$Arguments,
        [string]$LogPath
    )
    $stopwatch = [Diagnostics.Stopwatch]::StartNew()
    $exitCode = 1
    try {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ScriptPath @Arguments 2>&1 |
            Tee-Object -FilePath $LogPath |
            ForEach-Object { Write-Host $_ }
        $exitCode = $LASTEXITCODE
    } catch {
        $_ | Out-String | Tee-Object -FilePath $LogPath -Append | Write-Host
        $exitCode = 1
    } finally {
        $stopwatch.Stop()
    }
    return [pscustomobject]@{
        exit_code = $exitCode
        elapsed_seconds = $stopwatch.Elapsed.TotalSeconds
    }
}

$startedAt = Get-Date
$rows = [Collections.Generic.List[object]]::new()
$overallExitCode = 0

Write-Host "GeoFEM Case2 -> Case4 Mohr-Coulomb Numba validation"
Write-Host "Output: $runDirectory"
Write-Host "Case2: strict factor_tol=0.005 confirmation"
Write-Host "Case4: current speed-guarded Auto SRM"
Write-Host ""

if (-not $SkipCase2) {
    $case2Directory = Join-Path $runDirectory "case2"
    $case2Args = @("-OutputRoot", $case2Directory)
    if ($DryRun) {
        $case2Args += "-DryRun"
    }
    Write-Host "[Case2] starting"
    $case2Invocation = Invoke-NestedRunner $case2Runner $case2Args (Join-Path $runDirectory "case2_launcher.log")
    $case2StrictPath = Join-Path $case2Directory "strict_confirmation_summary.json"
    $case2SummaryPath = Join-Path $case2Directory "summary.json"
    $case2Metrics = Get-SrmMetrics $case2SummaryPath
    $case2Status = if (Test-Path -LiteralPath $case2StrictPath -PathType Leaf) {
        Get-PropertyValue (Get-Content -LiteralPath $case2StrictPath -Raw | ConvertFrom-Json) "status" "unknown"
    } elseif ($DryRun -and $case2Invocation.exit_code -eq 0) {
        "dry_run"
    } else {
        "missing_summary"
    }
    $rows.Add([pscustomobject][ordered]@{
        case = 2
        status = $case2Status
        exit_code = $case2Invocation.exit_code
        factor_of_safety = $case2Metrics.factor_of_safety
        stable_factor = $case2Metrics.stable_factor
        failed_factor = $case2Metrics.failed_factor
        trial_count = $case2Metrics.trial_count
        elapsed_seconds = [Math]::Round($case2Invocation.elapsed_seconds, 3)
        mc_numba_to_python_fallback_count = $case2Metrics.mc_numba_to_python_fallback_count
        mc_numba_regularized_projection_count = $case2Metrics.mc_numba_regularized_projection_count
        mc_regularized_projection_count = $case2Metrics.mc_regularized_projection_count
        output_directory = $case2Directory
    })
    if ($case2Invocation.exit_code -ne 0) {
        $overallExitCode = 1
        Write-Warning "Case2 did not finish cleanly. Case4 will still run so both validation outputs are collected."
    }
}

$case2CancelFile = Join-Path $runDirectory "case2\cancel.request"
if (-not $SkipCase4 -and -not (Test-Path -LiteralPath $case2CancelFile)) {
    $case4RunDirectory = Join-Path $runDirectory "case4_runner"
    $case4Args = @(
        "-Cases", "4",
        "-OutputRoot", $case4RunDirectory,
        "-ParallelPolicy", $ParallelPolicy,
        "-Workers", $Workers
    )
    if ($DryRun) {
        $case4Args += "-DryRun"
    }
    Write-Host ""
    Write-Host "[Case4] starting"
    $case4Invocation = Invoke-NestedRunner $case4Runner $case4Args (Join-Path $runDirectory "case4_launcher.log")
    $case4Directory = Join-Path $case4RunDirectory "case4"
    $case4SummaryPath = Join-Path $case4Directory "summary.json"
    $case4Metrics = Get-SrmMetrics $case4SummaryPath
    $case4Status = if ($DryRun -and $case4Invocation.exit_code -eq 0) {
        "dry_run"
    } elseif ($case4Invocation.exit_code -eq 0 -and (Test-Path -LiteralPath $case4SummaryPath -PathType Leaf)) {
        "completed"
    } else {
        "failed"
    }
    $rows.Add([pscustomobject][ordered]@{
        case = 4
        status = $case4Status
        exit_code = $case4Invocation.exit_code
        factor_of_safety = $case4Metrics.factor_of_safety
        stable_factor = $case4Metrics.stable_factor
        failed_factor = $case4Metrics.failed_factor
        trial_count = $case4Metrics.trial_count
        elapsed_seconds = [Math]::Round($case4Invocation.elapsed_seconds, 3)
        mc_numba_to_python_fallback_count = $case4Metrics.mc_numba_to_python_fallback_count
        mc_numba_regularized_projection_count = $case4Metrics.mc_numba_regularized_projection_count
        mc_regularized_projection_count = $case4Metrics.mc_regularized_projection_count
        output_directory = $case4Directory
    })
    if ($case4Invocation.exit_code -ne 0) {
        $overallExitCode = 1
    }
} elseif (Test-Path -LiteralPath $case2CancelFile) {
    Write-Warning "Cancellation was requested during Case2. Case4 was not started."
    $overallExitCode = 2
}

$finishedAt = Get-Date
$status = if ($DryRun) {
    "dry_run"
} elseif ($overallExitCode -eq 0) {
    "completed"
} elseif ($overallExitCode -eq 2) {
    "cancelled"
} else {
    "completed_with_failures"
}
$summary = [ordered]@{
    schema = "geofem.case2-4_mc_numba_validation.v1"
    status = $status
    started_at = $startedAt.ToString("o")
    finished_at = $finishedAt.ToString("o")
    elapsed_seconds = ($finishedAt - $startedAt).TotalSeconds
    workers = $Workers
    parallel_policy = $ParallelPolicy
    case2_yaml = Join-Path $case2Package "sustainability_2024_case2_quad4_sri_strict_fos_005.yaml"
    case4_yaml = Join-Path $case4Package "sustainability_2024_case4_quad4_sri_auto_srm_speed_guarded.yaml"
    cases = @($rows)
}
$summaryJson = Join-Path $runDirectory "case2-4_mc_numba_validation_summary.json"
$summaryCsv = Join-Path $runDirectory "case2-4_mc_numba_validation_summary.csv"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryJson -Encoding utf8
@($rows) | Export-Csv -LiteralPath $summaryCsv -NoTypeInformation -Encoding utf8

Write-Host ""
Write-Host "Validation status: $status"
Write-Host "Summary JSON: $summaryJson"
Write-Host "Summary CSV: $summaryCsv"
exit $overallExitCode
