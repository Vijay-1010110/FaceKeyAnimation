import os
import sys

def create_windows_shortcut(target_path, shortcut_path, icon_path=None, description=""):
    import subprocess
    target_path = os.path.abspath(target_path)
    shortcut_path = os.path.abspath(shortcut_path)
    working_dir = os.path.dirname(target_path)
    icon_line = f"$s.IconLocation = '{os.path.abspath(icon_path)}';" if icon_path and os.path.exists(icon_path) else ""
    
    ps_command = f"""
    $ws = New-Object -ComObject WScript.Shell;
    $s = $ws.CreateShortcut('{shortcut_path}');
    $s.TargetPath = '{target_path}';
    $s.WorkingDirectory = '{working_dir}';
    $s.Description = '{description}';
    {icon_line}
    $s.Save();
    """
    cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_command]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[OK] Created shortcut: {shortcut_path}")
    else:
        print(f"[ERROR] Failed to create shortcut {shortcut_path}: {res.stderr}")

def main():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.exists(desktop):
        # Fallback
        desktop = "C:\\Users\\Vijay\\Desktop"

    project_dir = os.path.abspath(os.path.dirname(__file__))
    icon_path = os.path.join(project_dir, "assets", "facekey_icon.ico")
    
    # 1. Master Control Launcher Shortcut (All options menu)
    launcher_bat = os.path.join(project_dir, "LAUNCHER.bat")
    shortcut_launcher = os.path.join(desktop, "FaceKey Animation Studio.lnk")
    create_windows_shortcut(
        target_path=launcher_bat,
        shortcut_path=shortcut_launcher,
        icon_path=icon_path,
        description="FaceKey Animation Studio - Master Interactive Launcher"
    )

    # 2. Quick Direct YouTube Studio (1-Click Start)
    quick_bat = os.path.join(project_dir, "LAUNCH_MONITOR_STUDIO.bat")
    shortcut_quick = os.path.join(desktop, "FaceKey Quick Studio (YouTube).lnk")
    create_windows_shortcut(
        target_path=quick_bat,
        shortcut_path=shortcut_quick,
        icon_path=icon_path,
        description="Quick Launch FaceKey YouTube Live Monitor Studio"
    )

    # 3. Stop Studio Peacefully (1-Click Stop & Save)
    stop_bat = os.path.join(project_dir, "STOP_STUDIO_PEACEFULLY.bat")
    shortcut_stop = os.path.join(desktop, "FaceKey Stop Peacefully.lnk")
    create_windows_shortcut(
        target_path=stop_bat,
        shortcut_path=shortcut_stop,
        icon_path=icon_path,
        description="Stop FaceKey Recording and Safely Flush Frames to Disk"
    )

    # 4. Switch Tab to YouTube Shortcut (1-Click Instant Tab Switch)
    switch_bat = os.path.join(project_dir, "SWITCH_TAB_TO_YOUTUBE.bat")
    shortcut_switch = os.path.join(desktop, "FaceKey Switch Tab (YouTube).lnk")
    create_windows_shortcut(
        target_path=switch_bat,
        shortcut_path=shortcut_switch,
        icon_path=icon_path,
        description="Immediately Switch FaceKey Capture to Active YouTube Video"
    )

    # 5. Native Window / App Selector GUI (Choose which window to capture)
    select_bat = os.path.join(project_dir, "SELECT_CAPTURE_WINDOW.bat")
    shortcut_select = os.path.join(desktop, "FaceKey Select Window or App.lnk")
    create_windows_shortcut(
        target_path=select_bat,
        shortcut_path=shortcut_select,
        icon_path=icon_path,
        description="Pick Any Open Window or Application to Capture in FaceKey Studio"
    )

if __name__ == "__main__":
    main()
