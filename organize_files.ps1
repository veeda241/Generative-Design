# Organize project structure
Set-Location "c:\hackathon\Gemini_CLI\Generative-Design"

# Create directories if they don't exist
New-Item -ItemType Directory -Force -Path "scripts"
New-Item -ItemType Directory -Force -Path "misc"

# Move Python scripts to scripts/
$pythonScripts = @(
    "check_ollama.py",
    "cleanup_aggressive.py", 
    "cleanup_ollama.py",
    "install_cuda_torch.py",
    "install_deps.py",
    "migrate_models.py",
    "preload_models.py",
    "robust_downloader.py",
    "robust_downloader_v2.py",
    "run_diagnostic.py",
    "stage_absolute.py",
    "stage_final.py",
    "stage_models.py",
    "test_cuda_init.py",
    "test_mkdir.py"
)

foreach ($file in $pythonScripts) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "scripts\" -Force
        Write-Host "Moved $file to scripts/"
    }
}

# Move shell scripts to scripts/
$shellScripts = @(
    "recover_system.ps1",
    "reorganize.ps1",
    "start.bat",
    "start.sh"
)

foreach ($file in $shellScripts) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "scripts\" -Force
        Write-Host "Moved $file to scripts/"
    }
}

# Move misc files
$miscFiles = @(
    "write_test.txt",
    "package-lock.json"
)

foreach ($file in $miscFiles) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "misc\" -Force
        Write-Host "Moved $file to misc/"
    }
}

Write-Host "`nOrganization complete!"
Write-Host "Listing root directory:"
Get-ChildItem -Path "." | Select-Object Name, Mode | Format-Table -AutoSize
