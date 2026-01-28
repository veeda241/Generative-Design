# Reorganize project structure
# Run this script from the project root: .\reorganize.ps1

$ErrorActionPreference = "SilentlyContinue"

Write-Host "Reorganizing Generative-Design project structure..." -ForegroundColor Cyan

# Create directories
Write-Host "Creating directories..."
New-Item -ItemType Directory -Path "docs" -Force | Out-Null
New-Item -ItemType Directory -Path "scripts" -Force | Out-Null
New-Item -ItemType Directory -Path "backend\src" -Force | Out-Null

# Move documentation files
Write-Host "Moving documentation files to docs/..."
$mdFiles = @(
    "00_START_HERE.md", "3D_BIM_INTEGRATION.md", "BEFORE_AFTER_COMPARISON.md",
    "COMPLETE_FILE_STRUCTURE.md", "COMPLETE_SETUP_GUIDE.md", "DEPLOYMENT_GUIDE.md",
    "DOCUMENTATION_INDEX.md", "IMPLEMENTATION_SUMMARY.md", "INDEX.md",
    "OPTIMIZATION_SUMMARY.md", "POINT_E_README.md", "POINT_E_SETUP.md",
    "PROJECT_COMPLETION_SUMMARY.md", "QUICK_REFERENCE.md", "QUICK_START.md",
    "QUICK_USAGE_GUIDE.md", "README_PROFESSIONAL_UI.md", "START_HERE.md",
    "TECH_STACK_VERIFIED.md", "TESTING_GUIDE.md", "TODAYS_UPDATES.md",
    "UI_IMPROVEMENTS.md", "UI_REDESIGN_COMPLETE.md", "VERIFICATION_CHECKLIST.md"
)
foreach ($file in $mdFiles) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "docs\" -Force
    }
}

# Move utility scripts
Write-Host "Moving utility scripts to scripts/..."
$scriptFiles = @(
    "check_ollama.py", "cleanup_aggressive.py", "cleanup_ollama.py",
    "install_cuda_torch.py", "install_deps.py", "migrate_models.py",
    "preload_models.py", "recover_system.ps1", "robust_downloader.py",
    "robust_downloader_v2.py", "run_diagnostic.py", "stage_absolute.py",
    "stage_final.py", "stage_models.py", "test_cuda_init.py", "test_mkdir.py"
)
foreach ($file in $scriptFiles) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "scripts\" -Force
    }
}

# Move backend source files
Write-Host "Moving backend source files to backend/src/..."
$backendFiles = @(
    "main.py", "engine.py", "local_engine.py", "point_e_service.py",
    "exporter.py", "bim_handler.py", "examples.py", "test_engine.py", "test_point_e.py"
)
foreach ($file in $backendFiles) {
    $path = "backend\$file"
    if (Test-Path $path) {
        Move-Item -Path $path -Destination "backend\src\" -Force
    }
}

# Create __init__.py for backend/src
"# Backend Source Package" | Out-File -FilePath "backend\src\__init__.py" -Encoding utf8

# Clean up temp files
if (Test-Path "write_test.txt") { Remove-Item "write_test.txt" -Force }

Write-Host ""
Write-Host "Done! New structure:" -ForegroundColor Green
Write-Host ""
Get-ChildItem -Directory | ForEach-Object { Write-Host "  $($_.Name)/" -ForegroundColor Yellow }
Get-ChildItem -File | ForEach-Object { Write-Host "  $($_.Name)" }
