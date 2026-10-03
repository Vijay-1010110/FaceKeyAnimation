@echo off
title FaceKey - Switch Capture to YouTube
cd /d "%~dp0"
"D:\python_3.11\python.exe" switch_tab.py youtube
timeout /t 2
