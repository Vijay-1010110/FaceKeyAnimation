"""Command/Script to trigger immediate tab switch in FaceKey Studio."""
import sys
import os
import time

def main():
    target = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    flag_path = os.path.join(script_dir, "switch.flag")

    with open(flag_path, "w", encoding="utf-8") as f:
        f.write(target)

    label = f"'{target}'" if target else "NEXT WINDOW / TAB"
    print("=" * 60)
    print(f" [OK] Signal sent: FaceKey Studio will switch to {label} immediately.")
    print("=" * 60)
    time.sleep(1)

if __name__ == "__main__":
    main()
