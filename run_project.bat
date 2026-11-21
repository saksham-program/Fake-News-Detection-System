@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title AI Powered Fake News Detection System

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY ( where python >nul 2>&1 && set "PY=python" )
if not defined PY (
    echo ERROR: Python 3.10+ was not found. Install it from python.org and tick "Add Python to PATH".
    goto :failure
)

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PY% -m venv venv || goto :failure
)

set "VENV_PY=venv\Scripts\python.exe"
echo Installing requirements...
"%VENV_PY%" -m pip install --quiet --upgrade pip || goto :failure
"%VENV_PY%" -m pip install --quiet -r requirements.txt || goto :failure

if not exist "model.pkl" ( "%VENV_PY%" train.py || goto :failure )
if not exist "vectorizer.pkl" ( "%VENV_PY%" train.py || goto :failure )

echo.
echo Starting server at http://127.0.0.1:5000
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5000"
"%VENV_PY%" app.py
goto :eof

:failure
echo.
echo PROJECT FAILED TO START
pause
exit /b 1
