@echo off
title FaceKey Silent 480p Batch YouTube Queue Runner (100h Milestone)
cd /d "%~dp0"
echo =========================================================================
echo   FaceKey Silent 480p Batch YouTube Queue Runner & Resume Engine
echo   - Watching: sessions\youtubeURLtoProcess.txt
echo   - 100%% Silent Headless Background Mode (0 GUI Window, Lowest CPU/GPU)
echo   - Add or paste YouTube links into that file anytime!
echo   - Automatically resumes from your save point if interrupted
echo   - Tracking Phase 1 Milestone: 100 Hours Target (~48 GB estimated)
echo   - Stop peacefully anytime by pressing Ctrl+C or running STOP_STUDIO_PEACEFULLY.bat
echo =========================================================================
"D:\python_3.11\python.exe" batch_stream_runner.py
if errorlevel 1 (
    echo.
    echo Batch processing paused or completed.
)
pause
