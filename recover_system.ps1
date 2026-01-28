$ErrorActionPreference = 'SilentlyContinue'
echo "Starting System Recovery..."

# Kill all python and pip
echo "Terminating Python and Pip..."
Get-Process python | Stop-Process -Force
Get-Process pip | Stop-Process -Force

# Wait for handles to release
Start-Sleep -Seconds 5

# Clear locks in backend cache
echo "Cleaning backend cache..."
$lockedFiles = Get-ChildItem -Path "backend\point_e_model_cache\*" -Include "*.lock", "*.tmp"
foreach ($f in $lockedFiles) {
    Remove-Item $f.FullName -Force
}

# Ensure new cache exists
$newCache = "c:\hackathon\Gemini_CLI\Generative-Design\point_e_cache_fixed"
echo "Target Cache: $newCache"
if (!(Test-Path $newCache)) {
    New-Item -ItemType Directory -Path $newCache -Force
}

# Final download of upsampler
echo "Downloading Upsampler..."
Invoke-WebRequest -Uri 'https://openaipublic.azureedge.net/main/point-e/upsample_40m.pt' -OutFile "$newCache\upsample_40m.pt"

echo "RECOVERY FINISHED"
