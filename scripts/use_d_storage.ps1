# Dot-source before project commands: . ./scripts/use_d_storage.ps1
$storageRoot = 'D:\DEMO\superembedding-storage'
$projectCacheSettings = @{
    SUPEREMBEDDING_STORAGE = $storageRoot
    HF_HOME = "$storageRoot\huggingface"
    HF_HUB_CACHE = "$storageRoot\huggingface\hub"
    HUGGINGFACE_HUB_CACHE = "$storageRoot\huggingface\hub"
    HF_DATASETS_CACHE = "$storageRoot\huggingface\datasets"
    HF_XET_CACHE = "$storageRoot\huggingface\xet"
    TORCH_HOME = "$storageRoot\torch"
    PIP_CACHE_DIR = "$storageRoot\pip-cache"
    UV_CACHE_DIR = "$storageRoot\uv-cache"
    TEMP = "$storageRoot\tmp"
    TMP = "$storageRoot\tmp"
    TMPDIR = "$storageRoot\tmp"
    XDG_CACHE_HOME = "$storageRoot\xdg-cache"
    MPLCONFIGDIR = "$storageRoot\matplotlib"
    NUMBA_CACHE_DIR = "$storageRoot\numba"
    TRITON_CACHE_DIR = "$storageRoot\triton"
    TORCH_EXTENSIONS_DIR = "$storageRoot\torch-extensions"
}
foreach ($entry in $projectCacheSettings.GetEnumerator()) {
    New-Item -ItemType Directory -Path $entry.Value -Force | Out-Null
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
}
