@echo off
title Facial MoCap & Single-Face Speaking Gate Verification Studio
cd /d "%~dp0"
echo =========================================================================
echo   Starting Facial MoCap & Single-Face Speaking Gate Verification Studio
echo =========================================================================
"D:\python_3.11\python.exe" test_face_speaker_tool.py
if errorlevel 1 (
    echo.
    echo An error occurred while running the studio.
)
pause
