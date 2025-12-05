# AI Tutor Development Scripts (PowerShell)
# For Windows users who prefer PowerShell over Make

Write-Host "AI Tutor - Development Helper" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host ""

function Show-Help {
    Write-Host "Available commands:" -ForegroundColor Yellow
    Write-Host "  .\scripts.ps1 setup      - Create venv and install dependencies"
    Write-Host "  .\scripts.ps1 dev        - Run development server"
    Write-Host "  .\scripts.ps1 test       - Run all tests"
    Write-Host "  .\scripts.ps1 docker-up  - Start with Docker"
    Write-Host "  .\scripts.ps1 docker-down- Stop Docker containers"
    Write-Host ""
}

function Setup-Environment {
    Write-Host "Creating virtual environment..." -ForegroundColor Green
    python -m venv venv
    
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    .\venv\Scripts\Activate.ps1
    
    Write-Host "Installing dependencies..." -ForegroundColor Green
    pip install -r requirements.txt
    
    Write-Host ""
    Write-Host "Setup complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Copy .env.example to .env"
    Write-Host "2. Add your GEMINI_API_KEY to .env"
    Write-Host "3. Run '.\scripts.ps1 dev' to start the server"
}

function Start-Dev {
    Write-Host "Starting development server..." -ForegroundColor Green
    .\venv\Scripts\Activate.ps1
    Set-Location src/backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
}

function Run-Tests {
    Write-Host "Running tests..." -ForegroundColor Green
    .\venv\Scripts\Activate.ps1
    pytest tests/ -v
}

function Docker-Up {
    Write-Host "Starting Docker containers..." -ForegroundColor Green
    docker-compose -f deploy/docker-compose.yml up --build
}

function Docker-Down {
    Write-Host "Stopping Docker containers..." -ForegroundColor Green
    docker-compose -f deploy/docker-compose.yml down
}

# Main command handler
$command = $args[0]

switch ($command) {
    "setup" { Setup-Environment }
    "dev" { Start-Dev }
    "test" { Run-Tests }
    "docker-up" { Docker-Up }
    "docker-down" { Docker-Down }
    default { Show-Help }
}
