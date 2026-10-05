"""Windows System Audio & Toast Notification Utility for FaceKey Studio."""

import os
import sys
import threading
import subprocess

def notify_user(title: str, message: str, sound: bool = True):
    """Deliver an immediate Windows audio chime and native Toast Notification.
    Works seamlessly in background or headless modes without blocking.
    """
    if sound and sys.platform == "win32":
        try:
            import winsound
            winsound.PlaySound("SystemNotification", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except Exception:
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            except Exception:
                pass

    def _toast_worker():
        # 1. Cloud / Webhook / Mobile Push Notification (Linux & Cross-Platform)
        ntfy_topic = os.environ.get("NTFY_TOPIC", "facekey_alerts_vijay")
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL") or os.environ.get("WEBHOOK_URL")

        # Send push notification to ntfy.sh (Zero-setup free push alerts)
        if ntfy_topic:
            try:
                import urllib.request
                req = urllib.request.Request(
                    f"https://ntfy.sh/{ntfy_topic}",
                    data=f"{title}: {message}".encode("utf-8"),
                    headers={"Title": title, "Priority": "high"}
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception:
                pass

        # Send webhook notification (e.g. Discord) if configured
        if webhook_url:
            try:
                import urllib.request
                import json
                payload = json.dumps({"content": f"🔔 **{title}**\n{message}"}).encode("utf-8")
                req = urllib.request.Request(
                    webhook_url,
                    data=payload,
                    headers={"Content-Type": "application/json", "User-Agent": "FaceKeyNotifier/1.0"}
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception:
                pass

        if sys.platform != "win32":
            return

        # 2. Windows Native Toast Notification
        clean_title = title.replace("'", "''").replace('"', '`"')
        clean_msg = message.replace("'", "''").replace('"', '`"')
        ps_code = f"""
        try {{
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
            $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
            $textNodes = $template.GetElementsByTagName('text')
            $textNodes.Item(0).AppendChild($template.CreateTextNode('{clean_title}')) | Out-Null
            $textNodes.Item(1).AppendChild($template.CreateTextNode('{clean_msg}')) | Out-Null
            $notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('FaceKey Animation Studio')
            $notification = [Windows.UI.Notifications.ToastNotification]::new($template)
            $notifier.Show($notification)
        }} catch {{
            # Silent fallback if Windows Toast service is restricted
        }}
        """
        try:
            subprocess.run(
                ["powershell", "-WindowStyle", "Hidden", "-Command", ps_code],
                creationflags=0x08000000 if os.name == "nt" else 0,
                timeout=12
            )
        except Exception:
            pass

    threading.Thread(target=_toast_worker, daemon=True).start()
