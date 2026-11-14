# Builds notebooks2/*.ipynb from notebooks2/src/[0-9]*.py (percent format) and copies nb2_common.py.
#   "# %%"             -> starts a code cell
#   "# %% [markdown]"  -> starts a markdown cell; its lines lose the leading "# " / "#"
# Usage: powershell -ExecutionPolicy Bypass -File notebooks2\build_notebooks.ps1
$ErrorActionPreference = 'Stop'
$root = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $root) { $root = "D:\research\New folder\Comparative approch of daga\Adaptive-Quantum-Kernel-Selection\notebooks2" }
$src = Join-Path $root 'src'
$utf8 = New-Object System.Text.UTF8Encoding($false)

Copy-Item (Join-Path $src 'nb2_common.py') (Join-Path $root 'nb2_common.py') -Force

function New-Cell($type, [string[]]$lines) {
    while ($lines.Count -gt 0 -and $lines[-1].Trim() -eq '') { $lines = $lines[0..($lines.Count - 2)] }
    while ($lines.Count -gt 0 -and $lines[0].Trim() -eq '') { $lines = if ($lines.Count -gt 1) { $lines[1..($lines.Count - 1)] } else { @() } }
    if ($lines.Count -eq 0) { return $null }
    $srcLines = New-Object System.Collections.ArrayList
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $l = $lines[$i]
        if ($i -lt $lines.Count - 1) { $l += "`n" }
        [void]$srcLines.Add($l)
    }
    $cell = [ordered]@{ cell_type = $type; metadata = @{}; source = [object[]]$srcLines.ToArray() }
    if ($type -eq 'code') { $cell.execution_count = $null; $cell.outputs = [object[]]@() }
    return $cell
}

Get-ChildItem $src | Where-Object { $_.Name -match '^\d{2}_.*\.py$' } | Sort-Object Name | ForEach-Object {
    $text = [IO.File]::ReadAllText($_.FullName, $utf8) -replace "`r`n", "`n"
    $cells = New-Object System.Collections.ArrayList
    $type = 'code'; $buf = New-Object System.Collections.ArrayList
    foreach ($line in ($text -split "`n")) {
        if ($line -match '^# %%') {
            $c = New-Cell $type ([string[]]$buf.ToArray()); if ($c) { [void]$cells.Add($c) }
            $buf.Clear(); $type = if ($line -match '\[markdown\]') { 'markdown' } else { 'code' }
            continue
        }
        if ($type -eq 'markdown') { $line = $line -replace '^# ?', '' }
        [void]$buf.Add($line)
    }
    $c = New-Cell $type ([string[]]$buf.ToArray()); if ($c) { [void]$cells.Add($c) }
    $nb = [ordered]@{
        cells = [object[]]$cells.ToArray()
        metadata = [ordered]@{
            kernelspec = [ordered]@{ display_name = 'Python 3'; language = 'python'; name = 'python3' }
            language_info = [ordered]@{ name = 'python' }
        }
        nbformat = 4; nbformat_minor = 5
    }
    $out = Join-Path $root ($_.BaseName + '.ipynb')
    [IO.File]::WriteAllText($out, ($nb | ConvertTo-Json -Depth 20), $utf8)
    Write-Output ("built {0}  ({1} cells)" -f (Split-Path $out -Leaf), $cells.Count)
}
