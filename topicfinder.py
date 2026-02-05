from pathlib import Path
from rosbags.highlevel import AnyReader

bagdir = Path(r"D:\Studium\BACHELOR ARBEIT\samson_data_reader")
bagfile = max(bagdir.glob("*.bag"), key=lambda p: p.stat().st_mtime)  # oder dein Finder
with AnyReader([bagfile]) as r:
    print("Topics in bag:")
    for c in r.connections:
        print(" ", c.topic, "->", c.msgtype)