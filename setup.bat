@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo   ViT Attention Rollout Explorer Setup
echo ==========================================
echo.

:: Create asset directories
if not exist "assets\images" mkdir "assets\images"
if not exist "assets\models" mkdir "assets\models"

:: Check conda is available
where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: conda not found. Make sure Miniconda is installed and added to PATH.
    echo        Try opening "Anaconda Prompt" instead of regular cmd.
    pause
    exit /b 1
)

:: Create conda environment if it doesn't exist
conda info --envs | findstr /C:"vit-explorer" >nul 2>&1
if errorlevel 1 (
    echo Creating conda env "vit-explorer" with Python 3.10...
    conda create -y -n vit-explorer python=3.10
    if errorlevel 1 (
        echo ERROR: Failed to create conda environment.
        pause
        exit /b 1
    )
) else (
    echo Conda env "vit-explorer" already exists, skipping creation.
)

echo.
echo Installing Python dependencies...
call conda run -n vit-explorer pip install ^
    fastapi==0.111.0 ^
    "uvicorn[standard]==0.30.1" ^
    torch==2.3.0 ^
    torchvision==0.18.0 ^
    transformers==4.41.2 ^
    Pillow==10.3.0 ^
    opencv-python==4.9.0.80 ^
    "numpy==1.26.4" ^
    python-multipart==0.0.9

if errorlevel 1 (
    echo ERROR: Python dependency installation failed.
    pause
    exit /b 1
)

echo.
echo Installing Node dependencies (frontend)...
cd frontend
call npm install
if errorlevel 1 (
    echo ERROR: npm install failed. Make sure Node.js is installed.
    pause
    exit /b 1
)
cd ..

echo.
echo ==========================================
echo           Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo   1. Add images  -^>  assets\images\
echo   2. Add models  -^>  assets\models\
echo.
echo   3. Start backend (Terminal 1):
echo      conda activate vit-explorer
echo      cd backend
echo      uvicorn main:app --reload --port 8000
echo.
echo   4. Start frontend (Terminal 2):
echo      cd frontend
echo      npm run dev
echo.
echo   5. Open http://localhost:5173
echo.
pause
