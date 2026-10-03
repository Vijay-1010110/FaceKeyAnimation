@echo off
title FaceKey - Stop Studio Peacefully
cd /d "%~dp0"
echo =========================================================================
echo   Sending Peaceful Stop Signal to FaceKey Motion Studio...
echo =========================================================================
"D:\python_3.11\python.exe" stop_studio.py
echo.
echo All active frames are now being safely flushed and saved to disk.
timeout /t 3
