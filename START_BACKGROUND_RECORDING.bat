@echo off
title FaceKey - Silent Background Tab / Window Recording
cd /d "%~dp0"
echo =========================================================================
echo   FaceKey Background Recording Studio (Zero On-Screen Interruption)
echo =========================================================================
echo.
echo   Select recording source:
echo     [1] Specific Window in Background (e.g. Brave / Chrome / YouTube)
echo     [2] Headless Screen Capture (Entire display, NO preview window on screen)
echo     [3] Vivo Y21 Phone Camera Stream (via Wi-Fi / IP Webcam / DroidCam)
echo.
set /p choice="Enter choice (1, 2, or 3) [Default 1]: "
if "%choice%"=="" set choice=1

if "%choice%"=="1" (
    set /p win_name="Enter window title keyword to capture (e.g. Brave, Chrome, YouTube) [Default Brave]: "
    if "%win_name%"=="" set win_name=Brave
    echo.
    echo [*] Starting Background Window capture for window matching: "%win_name%"
    echo [*] You can minimize or cover the window; recording continues in background!
    echo [*] Press Shift+Esc anywhere or type 'stop' in this console to save peacefully.
    echo.
    "D:\python_3.11\python.exe" test_face_speaker_tool.py --mode window --window "%win_name%" --headless
) else if "%choice%"=="2" (
    echo.
    echo [*] Starting Headless Screen Capture (no preview window will appear on screen)
    echo [*] Press Shift+Esc anywhere or type 'stop' in this console to save peacefully.
    echo.
    "D:\python_3.11\python.exe" test_face_speaker_tool.py --mode screen --roi 0,0,1920,1080 --headless
) else if "%choice%"=="3" (
    set /p phone_url="Enter Phone Stream URL (e.g. http://192.168.1.15:8080/video): "
    echo.
    echo [*] Connecting to Phone Camera: %phone_url%
    echo [*] Phone runs hardware H.264 at <4%% CPU - remains ice-cold while reading!
    echo [*] Press Shift+Esc anywhere or type 'stop' in this console to save peacefully.
    echo.
    "D:\python_3.11\python.exe" test_face_speaker_tool.py --mode stream --stream-url "%phone_url%"
)

pause
