# Makefile for AI Tutor project

.PHONY: help setup dev test lint clean docker-up docker-down

# Default target
help:
	@echo "AI Tutor - Available Commands"
	@echo "=============================="
	@echo "make setup      - Create virtual environment and install dependencies"
	@echo "make dev        - Run the development server"
	@echo "make test       - Run all tests"
	@echo "make lint       - Run linting checks"
	@echo "make clean      - Remove generated files"
	@echo "make docker-up  - Start with Docker Compose"
	@echo "make docker-down- Stop Docker containers"

# Setup development environment
setup:
	python -m venv venv
	.\venv\Scripts\activate.ps1; pip install -r requirements.txt
	@echo ""
	@echo "Setup complete! Don't forget to:"
	@echo "1. Copy .env.example to .env"
	@echo "2. Add your GEMINI_API_KEY to .env"
	@echo "3. Run 'make dev' to start the server"

# Run development server
dev:
	.\venv\Scripts\activate.ps1; cd src/backend; uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	.\venv\Scripts\activate.ps1; pytest tests/ -v

# Run tests with coverage
test-cov:
	.\venv\Scripts\activate.ps1; pytest tests/ -v --cov=src/backend --cov-report=html

# Lint code
lint:
	.\venv\Scripts\activate.ps1; flake8 src/ tests/
	.\venv\Scripts\activate.ps1; black --check src/ tests/

# Format code
format:
	.\venv\Scripts\activate.ps1; black src/ tests/

# Clean generated files
clean:
	Remove-Item -Recurse -Force -ErrorAction SilentlyContinue __pycache__
	Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .pytest_cache
	Remove-Item -Recurse -Force -ErrorAction SilentlyContinue htmlcov
	Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .coverage
	Remove-Item -Recurse -Force -ErrorAction SilentlyContinue *.egg-info
	Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force

# Docker commands
docker-up:
	docker-compose -f deploy/docker-compose.yml up --build

docker-down:
	docker-compose -f deploy/docker-compose.yml down

# Production build
build:
	docker build -t ai-tutor:latest -f deploy/Dockerfile .
