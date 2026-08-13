@echo off
REM Quick fix for pandas circular import error (Windows)

echo ==========================================
echo Fixing Pandas Circular Import Error
echo ==========================================

echo.
echo [1/4] Clearing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo Done: Cache cleared

echo.
echo [2/4] Clearing pip cache...
pip cache purge
echo Done: Pip cache cleared

echo.
echo [3/4] Reinstalling pandas...
pip uninstall pandas -y
pip install --no-cache-dir pandas
echo Done: Pandas reinstalled

echo.
echo [4/4] Testing pandas import...
python -c "import pandas as pd; print(f'Success! Pandas {pd.__version__} imported correctly')"

if %errorlevel% neq 0 (
    echo Failed. Trying deeper fix...
    echo.
    echo [Extra] Reinstalling numpy and pandas with specific versions...
    pip uninstall numpy pandas -y
    pip install --no-cache-dir numpy==1.26.4
    pip install --no-cache-dir pandas==2.2.0
    python -c "import pandas as pd; print(f'Success! Pandas {pd.__version__} imported correctly')"
)

echo.
echo ==========================================
echo Fix complete!
echo ==========================================
echo.
echo If you're in a Jupyter notebook:
echo   1. Run this script
echo   2. Restart your kernel
echo   3. Re-import pandas
pause
