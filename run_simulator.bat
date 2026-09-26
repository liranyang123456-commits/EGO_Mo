@echo off
title EGO_Mo Simulation Dataset Generator
chcp 65001 >nul
cd /d "%~dp0"
set "PY=D:\anaconda\python.exe"
if not exist "%PY%" (
    echo Python not found: %PY%
    pause
    exit /b 1
)
"%PY%" -u "%~dp0run_simulator.py"
if errorlevel 1 pause
