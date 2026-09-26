@echo off
title EGO_Mo 采集
chcp 65001 >nul
cd /d "%~dp0"

if not exist "%~dp0datasets" mkdir "%~dp0datasets"
set "LOG=%~dp0datasets\last_launch.log"
echo [%date% %time%] 启动 >> "%LOG%"
echo cwd=%cd% >> "%LOG%"

set "PY="
if exist "D:\anaconda\python.exe" set "PY=D:\anaconda\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"

if not defined PY (
    echo 找不到 Anaconda 的 python.exe
    echo 本机应在 D:\anaconda\python.exe
    echo [%date% %time%] 无 python >> "%LOG%"
    pause
    exit /b 1
)

echo python=%PY% >> "%LOG%"
echo 使用 %PY%
echo.

"%PY%" -u "%~dp0capture_app.py"
set "ERR=%ERRORLEVEL%"
echo [%date% %time%] 退出码 %ERR% >> "%LOG%"
if not "%ERR%"=="0" (
    echo.
    echo 启动失败，错误码 %ERR%
    echo 日志: %LOG%
    pause
    exit /b %ERR%
)
