# Run only after migrate_storage_to_d.py reports fully verified copies.
$ErrorActionPreference = 'Stop'
$storageRoot = 'D:\DEMO\superembedding-storage'
$auditPath = Join-Path $storageRoot 'migration_verified.json'
$audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
if ($audit.state -ne 'verified') { throw 'All copies must pass SHA256 verification first' }
$allowed = @{
    'C:\Users\123\.cache\superembedding' = "$storageRoot\superembedding"
    'C:\Users\123\.cache\huggingface' = "$storageRoot\huggingface"
    'C:\Users\123\AppData\Local\pip\Cache' = "$storageRoot\pip-cache"
    'C:\Users\123\AppData\Local\uv\cache' = "$storageRoot\uv-cache"
}
$results = @()
$completedPath = Join-Path $storageRoot 'migration_completed.json'
if (Test-Path -LiteralPath $completedPath) {
    $previous = Get-Content -LiteralPath $completedPath -Raw | ConvertFrom-Json
    $results += @($previous.roots)
}
foreach ($entry in $audit.roots) {
    $source = [IO.Path]::GetFullPath($entry.source)
    $destination = [IO.Path]::GetFullPath($entry.destination)
    if (-not $allowed.ContainsKey($source) -or $allowed[$source] -ne $destination) {
        throw 'Migration root is outside the approved exact path mapping'
    }
    if (-not $entry.all_sha256_match -or -not $entry.source_stable) { throw 'Unverified migration root' }
    $item = Get-Item -LiteralPath $source -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Source must still be a real directory' }
    if (-not (Test-Path -LiteralPath $destination -PathType Container)) { throw 'Destination missing' }
    $preserved = $source + '.verified-migration-original'
    if (Test-Path -LiteralPath $preserved) { throw 'Previous preserved source exists; inspect before continuing' }
    # Same-volume rename; deletion is limited to this exact preserved source.
    Move-Item -LiteralPath $source -Destination $preserved
    try {
        New-Item -ItemType Junction -Path $source -Target $destination | Out-Null
        $junction = Get-Item -LiteralPath $source -Force
        if ($junction.LinkType -ne 'Junction' -or [IO.Path]::GetFullPath($junction.Target) -ne $destination) {
            throw 'Compatibility junction verification failed; original preserved'
        }
    } catch {
        throw "Directory redirection failed; preserved original remains at $preserved. $($_.Exception.Message)"
    }
    $deleteItem = Get-Item -LiteralPath $preserved -Force
    $resolved = (Resolve-Path -LiteralPath $preserved).Path
    if ($resolved -ne $source + '.verified-migration-original' -or
        ($deleteItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
        [IO.Path]::GetPathRoot($resolved) -ne 'C:\') {
        throw 'Preserved source deletion path validation failed'
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
    $results += [PSCustomObject]@{source=$source;destination=$destination;files=$entry.files;bytes=$entry.bytes;sha256_verified=$true;junction_verified=$true;source_copy_removed=$true}
    [PSCustomObject]@{migrated=$source;destination=$destination;freed_bytes=$entry.bytes} | ConvertTo-Json -Compress
}
@{state='migrated';roots=$results;total_bytes=($results | Measure-Object -Property bytes -Sum).Sum;completed_utc=[DateTime]::UtcNow.ToString('o')} |
    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $storageRoot 'migration_completed.json') -Encoding utf8

. "$PSScriptRoot\use_d_storage.ps1"
# Persist only cache variables at user scope. System-wide TEMP is not changed.
foreach ($name in @('SUPEREMBEDDING_STORAGE','HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','HF_DATASETS_CACHE','HF_XET_CACHE','TORCH_HOME','PIP_CACHE_DIR','UV_CACHE_DIR')) {
    [Environment]::SetEnvironmentVariable($name,[Environment]::GetEnvironmentVariable($name,'Process'),'User')
}
