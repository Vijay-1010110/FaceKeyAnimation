"""Signal script to gracefully and peacefully stop FaceKey Studio from any terminal."""
import os
import sys

flag_file = os.path.join(os.path.dirname(__file__), "stop.flag")
with open(flag_file, "w") as f:
    f.write("STOP")

print("=" * 70)
print(" [STOP SIGNAL SENT]")
print(" FaceKey Studio is now peacefully finalizing and saving all recorded frames to disk.")
print("=" * 70)
