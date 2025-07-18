@echo off
echo Resume Scoring Agent - Windows Setup
echo =====================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found, proceeding with setup...

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
python -m pip install -r requirements.txt

REM Run setup script
echo Running setup script...
python setup.py

REM Create desktop shortcut (optional)
set /p create_shortcut="Create desktop shortcut? (y/n): "
if /i "%create_shortcut%"=="y" (
    echo Creating desktop shortcut...
    echo @echo off > "%USERPROFILE%\Desktop\Resume Scoring Agent.bat"
    echo cd /d "%CD%" >> "%USERPROFILE%\Desktop\Resume Scoring Agent.bat"
    echo streamlit run app.py >> "%USERPROFILE%\Desktop\Resume Scoring Agent.bat"
    echo pause >> "%USERPROFILE%\Desktop\Resume Scoring Agent.bat"
    echo Desktop shortcut created!
)

echo.
echo Setup complete! 
echo.
echo To run the application:
echo   streamlit run app.py
echo.
echo Or double-click the desktop shortcut if created.
echo.
pause
