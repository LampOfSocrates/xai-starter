<#
.SYNOPSIS
    Execute every lesson notebook top-to-bottom, in dependency order.

.DESCRIPTION
    Runs each notebook with `jupyter nbconvert --execute`, writing the
    executed copy (with outputs) back in place. Order follows README.md.

    - Times each notebook; prints the time AND appends it to a log file.
    - Points every HuggingFace cache at ONE shared directory so datasets and
      models are downloaded once by the first notebook and reused by the rest.
    - Stops on the first failure unless -ContinueOnError is given.

.EXAMPLE
    .\run_all_notebooks.ps1
    .\run_all_notebooks.ps1 -ContinueOnError -TimeoutSec 3600
    .\run_all_notebooks.ps1 -Tracks gnn,capstones
    .\run_all_notebooks.ps1 -CacheDir D:\hf_cache
#>
param(
    # Which tracks to run, in order. Default: all.
    [ValidateSet('plm', 'gnn', 'capstones', 'ig')]
    [string[]]$Tracks = @('plm', 'gnn', 'capstones', 'ig'),

    # Per-cell execution timeout in seconds.
    [int]$TimeoutSec = 1800,

    # Shared HuggingFace cache (datasets + models). Reused across all notebooks.
    [string]$CacheDir = (Join-Path $HOME '.cache\huggingface'),

    # Keep going after a notebook fails instead of aborting.
    [switch]$ContinueOnError
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# ---- Pin execution to the project venv -------------------------------------
# Bare `jupyter` resolves via PATH and can launch the wrong interpreter (one
# without torch/transformers). Always drive nbconvert through the venv python.
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    throw "venv python not found at $venvPython. Create it and install the deps (see README)."
}

# An activated conda env exports SSL_CERT_FILE pointing at conda's CA bundle.
# Our interpreter is the venv (not conda), so that path is wrong and httpx
# (huggingface downloads) fails with FileNotFoundError. Force the venv certifi.
$env:SSL_CERT_FILE = (& $venvPython -c 'import certifi; print(certifi.where())')
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE
Remove-Item Env:SSL_CERT_DIR -ErrorAction SilentlyContinue

# ---- Shared cache: download datasets & models ONCE, reuse everywhere -------
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
$env:HF_HOME = $CacheDir
$env:HF_DATASETS_CACHE = Join-Path $CacheDir 'datasets'
$env:TRANSFORMERS_CACHE = Join-Path $CacheDir 'hub'
$env:HF_HUB_DISABLE_TELEMETRY = '1'

# ---- Log file (timestamped) -----------------------------------------------
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $root "run_all_$stamp.log"

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

# Notebooks per track, in the order README says they build on each other.
$order = [ordered]@{
    plm       = @(
        'plm/plm_l1_embeddings_probe.ipynb'
        'plm/plm_l2_zero_shot_variants.ipynb'
        'plm/plm_l3_finetune_classification.ipynb'
        'plm/plm_l4_token_classification.ipynb'
        'plm/plm_l5_model_comparison.ipynb'
        'plm/plm_l6_attention_contacts.ipynb'
        'plm/plm_l7_lora_peft.ipynb'
        'plm/plm_l8_structure_aware.ipynb'
        'plm/plm_l9_embedding_retrieval.ipynb'
        'plm/plm_l10_inverse_folding.ipynb'
        'plm/plm_l11_calibration.ipynb'
    )
    gnn       = @(
        'gnn/gnn_l1_graphs_from_proteins.ipynb'
        'gnn/gnn_l2_node_classification.ipynb'
        'gnn/gnn_l3_graph_classification.ipynb'
        'gnn/gnn_l4_plm_plus_gnn.ipynb'
        'gnn/gnn_l5_equivariant_gnn.ipynb'
        'gnn/gnn_l6_real_structures.ipynb'
        'gnn/gnn_l7_edge_features.ipynb'
        'gnn/gnn_l8_contact_prediction.ipynb'
        'gnn/gnn_l9_knn_graphs.ipynb'
        'gnn/gnn_l10_oversmoothing.ipynb'
        'gnn/gnn_l11_interaction_graphs.ipynb'
    )
    capstones = @(
        'capstones/capstone_l1_end_to_end_plm_gnn.ipynb'
        'capstones/capstone_l2_benchmark_suite.ipynb'
    )
    ig        = @(
        'common/ig_l1_simple.ipynb'
        'common/ig_l2_tiny_network.ipynb'
    )
}

$queue = foreach ($t in $Tracks) { $order[$t] }

Log "Running $($queue.Count) notebook(s): tracks = $($Tracks -join ', ')"
Log "Shared HF cache: $CacheDir  (datasets & models downloaded once, reused)"
Log "Log file: $logFile"

$results = @()
$i = 0
$runStart = [System.Diagnostics.Stopwatch]::StartNew()
foreach ($nb in $queue) {
    $i++
    $path = Join-Path $root $nb
    if (-not (Test-Path $path)) {
        Log "[$i/$($queue.Count)] MISSING  $nb"
        $results += [pscustomobject]@{ Notebook = $nb; Status = 'missing'; Seconds = 0 }
        continue
    }

    Log "[$i/$($queue.Count)] RUN      $nb"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $venvPython -m nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.timeout=$TimeoutSec `
        --ExecutePreprocessor.kernel_name=python3 `
        $path
    $ok = $?
    $sw.Stop()
    $secs = [math]::Round($sw.Elapsed.TotalSeconds, 1)

    if ($ok) {
        Log "    OK       $nb  ($secs s)"
        $results += [pscustomobject]@{ Notebook = $nb; Status = 'ok'; Seconds = $secs }
    }
    else {
        Log "    FAILED   $nb  ($secs s)"
        $results += [pscustomobject]@{ Notebook = $nb; Status = 'failed'; Seconds = $secs }
        if (-not $ContinueOnError) {
            Log "Aborting (use -ContinueOnError to keep going)."
            break
        }
    }
}
$runStart.Stop()

Log ""
Log "===== Summary (total $([math]::Round($runStart.Elapsed.TotalSeconds, 1)) s) ====="
foreach ($r in $results) {
    Log ("  {0,-9} {1,8:n1}s  {2}" -f $r.Status, $r.Seconds, $r.Notebook)
}
$results | Format-Table -AutoSize

if ($results.Status -contains 'failed') { exit 1 }
