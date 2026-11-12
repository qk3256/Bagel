# BAGEL profiler traces

This directory contains the compressed PyTorch profiler traces collected during
the single-A100 BAGEL optimization experiments.

## Archive

- Reconstructed file: `bagel_pt_traces_20261107.tar.gz`
- Git files: `parts/bagel_pt_traces_20261107.tar.gz.part01` through `part20`
- Contents: 17 `*.pt.trace.json` files with their experiment directories
- Uncompressed trace bytes: 6,503,869,249
- Compressed bytes: 492,772,044
- SHA-256: `8325EF35EE198FE41E2363A3EBF2969353C0FF7A075028C844E59CD451F3CF59`
- Storage: twenty regular Git objects, each at most 25,000,000 bytes

Reassemble and extract from the repository root on Linux/macOS/Git Bash:

```bash
cat artifacts/traces/parts/bagel_pt_traces_20261107.tar.gz.part* > artifacts/traces/bagel_pt_traces_20261107.tar.gz
tar -xzf artifacts/traces/bagel_pt_traces_20261107.tar.gz
```

Reassemble on PowerShell, then extract:

```powershell
$output = [System.IO.File]::Create((Join-Path $PWD 'artifacts/traces/bagel_pt_traces_20261107.tar.gz'))
try {
  Get-ChildItem artifacts/traces/parts/*.part* | Sort-Object Name | ForEach-Object {
    $input = [System.IO.File]::OpenRead($_.FullName)
    try { $input.CopyTo($output) } finally { $input.Dispose() }
  }
} finally { $output.Dispose() }
(Get-FileHash artifacts/traces/bagel_pt_traces_20261107.tar.gz -Algorithm SHA256).Hash
tar -xzf artifacts/traces/bagel_pt_traces_20261107.tar.gz
```

The expected whole-archive hash is listed above. Per-part hashes are in
`parts/SHA256SUMS`.
