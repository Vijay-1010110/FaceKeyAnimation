@echo off
title FaceKey Live MoCap - YouTube Video Capture Studio
cd /d "%~dp0"
echo =========================================================================
echo   Starting FaceKey Motion Capture for YouTube / Live Screen Video
echo   - Native 1280x720 HD Preview with Razor-Sharp Overlay
echo   - Can be freely minimized to the taskbar while recording in background
echo   - Press [CROP ROI] or [R] in the window to drag a box around YouTube
echo   - Close peacefully via [X], [CLOSE], Shift+Esc, or STOP_STUDIO_PEACEFULLY.bat
echo =========================================================================
"D:\python_3.11\python.exe" test_face_speaker_tool.py --mode screen --roi 0,0,1920,1080
if errorlevel 1 (
    echo.
    echo An error occurred while running the studio.
)
pause
