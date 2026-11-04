[CmdletBinding()]
param(
    [ValidateSet("dashboard", "train", "docker")]
    [string]$Target = "dashboard"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Target -eq "docker") {
    $dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerCommand) {
        throw "Docker was not found. Install Docker Desktop, then reopen PowerShell."
    }

    & $dockerCommand.Source info *> $null
    if ($LASTEXITCODE -ne 0) {
        $desktopPath = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
        if (-not (Test-Path $desktopPath)) {
            throw "Docker Desktop is installed but its engine is not running. Start Docker Desktop and wait for it to finish starting."
        }

        Write-Host "Starting Docker Desktop; waiting for its engine..."
        Start-Process -FilePath $desktopPath
        $engineReady = $false
        for ($attempt = 0; $attempt -lt 60; $attempt++) {
            Start-Sleep -Seconds 2
            & $dockerCommand.Source info *> $null
            if ($LASTEXITCODE -eq 0) {
                $engineReady = $true
                break
            }
        }

        if (-not $engineReady) {
            throw "Docker Desktop did not become ready within 2 minutes. Check Docker Desktop, then rerun this script."
        }
    }

    Write-Host "Building and starting the dashboard and its services. Build progress will appear below; the first run can take several minutes."
    & $dockerCommand.Source compose --progress plain up --build -d
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose failed. Run 'docker compose logs --tail 100 brain-core' for details."
    }

    & $dockerCommand.Source compose ps
    Write-Host "Dashboard: http://localhost:8000"
    Write-Host "To stop the services, run: docker compose down"
    exit 0
}

$pythonExecutable = $null
$pythonPrefix = @()
$launcher = Get-Command py -ErrorAction SilentlyContinue

if ($launcher) {
    & $launcher.Source -3.11 --version *> $null
    if ($LASTEXITCODE -eq 0) {
        $pythonExecutable = $launcher.Source
        $pythonPrefix = @("-3.11")
    }
}

if (-not $pythonExecutable) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source -notmatch "\\WindowsApps\\") {
        $versionOutput = & $pythonCommand.Source --version 2>&1
        if ($LASTEXITCODE -eq 0 -and "$versionOutput" -match "^Python 3\.11\.") {
            $pythonExecutable = $pythonCommand.Source
        }
    }
}

if (-not $pythonExecutable) {
    throw "Python 3.11 was not found. Install Python 3.11 x64 from https://www.python.org/downloads/windows/ and enable 'Add python.exe to PATH', then reopen PowerShell."
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    & $pythonExecutable @pythonPrefix -m venv (Join-Path $PSScriptRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the project's .venv virtual environment."
    }
}

& $venvPython -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed. Check the pip output above and try again."
}

if ($Target -eq "train") {
    & $venvPython (Join-Path $PSScriptRoot "train.py")
} else {
    Write-Host "Dashboard: http://localhost:8000 (press Ctrl+C to stop)"
    & $venvPython (Join-Path $PSScriptRoot "dashboard_app.py") --port 8000 --no-tui
}

if ($LASTEXITCODE -ne 0) {
    throw "The $Target process exited with code $LASTEXITCODE."
}
