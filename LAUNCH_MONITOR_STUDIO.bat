@echo off
title FaceKey 1280x720 HD Live Monitor Studio
cd /d "%~dp0"
echo =========================================================================
echo   Starting FaceKey 1280x720 HD Live Monitor Studio
echo   - Interactive preview window will pop up on your screen
echo   - Click [SWITCH TAB] (or press 'Tab' or 'F9') to switch your YouTube tab
echo   - You can minimize freely with [_] to keep recording in background
echo   - Close peacefully with [X], [CLOSE], or Shift+Esc anytime
echo =========================================================================
"D:\python_3.11\python.exe" test_face_speaker_tool.py --mode window --window "Brave"
if errorlevel 1 (
    echo.
    echo An error occurred while running the studio.
)
pause
