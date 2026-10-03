"""Native Windows GUI Window & Tab Selector for FaceKey Studio.
Allows user to visually pick any open window, browser tab, or app to capture.
"""

import os
import sys

# Guarantee project root is in sys.path so 'src' can always be imported
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tkinter as tk
from tkinter import ttk, messagebox
import ctypes

def get_open_windows():
    """Enumerate all visible user windows, prioritizing video & browser windows."""
    from src.core.capture import WindowCaptureSource
    windows = WindowCaptureSource.list_all_windows()
    # Filter out helper windows
    filtered = []
    for h, t in windows:
        tl = t.lower()
        if any(ig in tl for ig in ("facekey", "drag box", "program manager", "realtek", "nvidia", "default ime")):
            continue
        filtered.append((h, t))
    return filtered

class WindowPickerApp:
    def __init__(self, root, on_select_callback=None):
        self.root = root
        self.on_select_callback = on_select_callback
        self.root.title("FaceKey Studio - Select Window / App to Capture")
        self.root.geometry("640x520")
        self.root.minsize(500, 380)
        self.root.configure(bg="#0f172a") # Slate 900

        # Try to set top-most initially so user sees it right away
        try:
            self.root.attributes("-topmost", True)
            self.root.after(500, lambda: self.root.attributes("-topmost", False))
        except Exception:
            pass

        self._build_ui()
        self.refresh_list()

    def _build_ui(self):
        # Header Frame
        header = tk.Frame(self.root, bg="#1e293b", pady=12, padx=16)
        header.pack(fill="x", side="top")

        lbl_title = tk.Label(
            header,
            text="🎯 Choose Application or Window to Capture",
            font=("Segoe UI", 13, "bold"),
            fg="#38bdf8",
            bg="#1e293b"
        )
        lbl_title.pack(anchor="w")

        lbl_sub = tk.Label(
            header,
            text="Click any open window below. FaceKey Studio will instantly lock capture onto it in background.",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b"
        )
        lbl_sub.pack(anchor="w", pady=(2, 0))

        tip_frame = tk.Frame(header, bg="#0f172a", padx=8, pady=4, relief="groove", borderwidth=1)
        tip_frame.pack(anchor="w", fill="x", pady=(6, 0))
        lbl_tip = tk.Label(
            tip_frame,
            text="💡 Tip: Drag your YouTube tab out into its own window to record it in the background while browsing other tabs freely!",
            font=("Segoe UI", 8, "bold"),
            fg="#facc15",
            bg="#0f172a"
        )
        lbl_tip.pack(anchor="w")

        # Windows Listbox Frame with scrollbar
        list_frame = tk.Frame(self.root, bg="#0f172a", padx=16, pady=12)
        list_frame.pack(fill="both", expand=True)

        self.scrollbar = tk.Scrollbar(list_frame, orient="vertical")
        self.listbox = tk.Listbox(
            list_frame,
            yscrollcommand=self.scrollbar.set,
            font=("Segoe UI", 10),
            bg="#1e293b",
            fg="#f8fafc",
            selectbackground="#0284c7",
            selectforeground="#ffffff",
            activestyle="none",
            borderwidth=1,
            relief="solid",
            highlightthickness=0,
            height=12
        )
        self.scrollbar.config(command=self.listbox.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True)

        self.listbox.bind("<Double-Button-1>", lambda event: self.select_current())
        self.listbox.bind("<Return>", lambda event: self.select_current())

        # Buttons Frame
        btn_frame = tk.Frame(self.root, bg="#0f172a", padx=16, pady=12)
        btn_frame.pack(fill="x", side="bottom")

        # Capture Selected Button
        self.btn_select = tk.Button(
            btn_frame,
            text="✅ Lock Capture onto Selected Window",
            font=("Segoe UI", 10, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            activebackground="#0369a1",
            activeforeground="#ffffff",
            padx=14,
            pady=6,
            relief="flat",
            cursor="hand2",
            command=self.select_current
        )
        self.btn_select.pack(side="left", padx=(0, 8))

        # Full Screen ROI Button
        self.btn_screen = tk.Button(
            btn_frame,
            text="🖥️ Capture Full Screen ROI",
            font=("Segoe UI", 9),
            bg="#334155",
            fg="#f8fafc",
            activebackground="#475569",
            activeforeground="#ffffff",
            padx=10,
            pady=6,
            relief="flat",
            cursor="hand2",
            command=self.select_screen_roi
        )
        self.btn_screen.pack(side="left", padx=(0, 8))

        # Refresh List Button
        self.btn_refresh = tk.Button(
            btn_frame,
            text="🔄 Refresh List",
            font=("Segoe UI", 9),
            bg="#1e293b",
            fg="#94a3b8",
            activebackground="#334155",
            activeforeground="#f8fafc",
            padx=10,
            pady=6,
            relief="flat",
            cursor="hand2",
            command=self.refresh_list
        )
        self.btn_refresh.pack(side="left")

        # Close Button
        self.btn_close = tk.Button(
            btn_frame,
            text="Cancel",
            font=("Segoe UI", 9),
            bg="#1e293b",
            fg="#ef4444",
            activebackground="#450a0a",
            activeforeground="#f87171",
            padx=10,
            pady=6,
            relief="flat",
            cursor="hand2",
            command=self.root.destroy
        )
        self.btn_close.pack(side="right")

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        self.windows_data = get_open_windows()

        if not self.windows_data:
            self.listbox.insert(tk.END, "  [No eligible windows found - Click Refresh or open a browser]")
            return

        for idx, (hwnd, title) in enumerate(self.windows_data):
            tl = title.lower()
            tag = "  "
            if any(k in tl for k in ("youtube", "video", "movie", "watch", "drishyam")):
                tag = "🎬 [VIDEO/YOUTUBE] "
            elif any(k in tl for k in ("brave", "chrome", "edge", "firefox", "opera")):
                tag = "🌐 [BROWSER] "
            
            self.listbox.insert(tk.END, f"{tag}{title}  (HWND: {hwnd})")

        # Select first entry by default
        self.listbox.selection_set(0)
        self.listbox.activate(0)

    def select_current(self):
        sel = self.listbox.curselection()
        if not sel or not hasattr(self, "windows_data") or not self.windows_data:
            return
        idx = sel[0]
        if idx < len(self.windows_data):
            hwnd, title = self.windows_data[idx]
            self._apply_switch(str(hwnd), title)

    def select_screen_roi(self):
        self._apply_switch("screen", "Full Desktop Screen ROI")

    def _apply_switch(self, target_identifier: str, target_title: str):
        # 1. Write switch.flag so any running FaceKey Studio picks it up immediately
        flag_file = os.path.join(project_root, "switch.flag")
        try:
            with open(flag_file, "w", encoding="utf-8") as f:
                f.write(target_identifier)
        except Exception:
            pass

        # 2. Call local callback if provided
        if self.on_select_callback:
            try:
                self.on_select_callback(target_identifier, target_title)
            except Exception:
                pass

        print(f"[OK] Switched FaceKey capture target to: [{target_identifier}] '{target_title}'")
        self.root.destroy()


def show_window_picker(on_select_callback=None):
    root = tk.Tk()
    app = WindowPickerApp(root, on_select_callback=on_select_callback)
    root.mainloop()

if __name__ == "__main__":
    # Ensure current thread desktop
    if sys.platform == "win32":
        try:
            user32 = ctypes.windll.user32
            hdesk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
            if not hdesk:
                hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
        except Exception:
            pass
    show_window_picker()
