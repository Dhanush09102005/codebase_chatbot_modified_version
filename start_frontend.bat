@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   Code Q&A Assistant - Frontend Setup
echo ========================================
echo.

REM Check if Node.js is installed
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Node.js is not installed. Please install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)

echo [OK] Node.js detected
node -v

REM Navigate to frontend directory
cd frontend
if %ERRORLEVEL% NEQ 0 (
    echo Frontend directory not found
    pause
    exit /b 1
)

REM Check if node_modules exists
if not exist "node_modules" (
    echo.
    echo Installing dependencies...
    call npm install
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
) else (
    echo [OK] Dependencies already installed
)

REM Check if .env exists
if not exist ".env" (
    echo.
    echo Creating .env file...
    copy .env.example .env
    echo [OK] .env file created
) else (
    echo [OK] .env file exists
)

echo.
echo ========================================
echo Setup complete! Starting development server...
echo ========================================
echo.
echo Frontend will be available at: http://localhost:3000
echo Make sure backend is running at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start development server
call npm run dev

pause
