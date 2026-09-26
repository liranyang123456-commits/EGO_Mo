@echo off
title EGO_Mo Capture
cd /d "%~dp0"

if not exist "%~dp0datasets" mkdir "%~dp0datasets"
set "LOG=%~dp0datasets\last_launch.log"
echo [%date% %time%] start >> "%LOG%"
echo cwd=%cd% >> "%LOG%"

set "PY="
if exist "D:\anaconda\python.exe" set "PY=D:\anaconda\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"

if not defined PY (
    echo Anaconda python.exe not found. Expected D:\anaconda\python.exe
    echo [%date% %time%] python not found >> "%LOG%"
    pause
    exit /b 1
)

echo python=%PY% >> "%LOG%"
echo Using %PY%
echo.

"%PY%" -u "%~dp0capture_app.py"
set "ERR=%ERRORLEVEL%"
echo [%date% %time%] exit code %ERR% >> "%LOG%"
if not "%ERR%"=="0" (
    echo.
    echo Capture app failed with exit code %ERR%
    echo Log: %LOG%
    pause
    exit /b %ERR%
)
