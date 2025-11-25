@echo off
echo ========================================
echo   HYBRID MAZE SOLVER - QUICK START
echo ========================================
echo.

echo [1/3] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

echo [2/3] Running tests...
python test_system.py
if errorlevel 1 (
    echo WARNING: Some tests failed
    echo.
)

echo [3/3] Running main solver...
echo.
python main.py --use-local-search
echo.

echo ========================================
echo Check the 'results' folder for outputs!
echo ========================================
pause
