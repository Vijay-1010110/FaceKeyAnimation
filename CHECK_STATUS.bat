@echo off
title FaceKey Studio — Live Status Check
cd /d "%~dp0"
"D:\python_3.11\python.exe" scripts\check_status.py
echo.
pause
