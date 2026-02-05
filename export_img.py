import os
import cv2
from samson_data_reader.data_reader import DataReader
from samson_data_reader.datacalsses import ImageData

# Pfad zu deinem Datenordner (wo SAMSON3.rec / SAMSON4.rec liegen)
DATA_DIR = "D:\Studium\BACHELOR ARBEIT\samson_data_reader"

# Ausgabeordner für exportierte Bilder
OUT_DIR = os.path.join(DATA_DIR, "exported_images_rotated")
os.makedirs(OUT_DIR, exist_ok=True)

# DataReader starten (Mono reicht)
dr = DataReader(
    path_data_dir=DATA_DIR,
    stereo_images=False,
    load_dvso=False, load_gps=False, load_lidar=False,
    image_size=None  # Originalgröße
)

for item in dr:
    if isinstance(item, ImageData):
        filename = f"{int(item.image_id):06d}.jpg"  # z.B. 000001.jpg
        out_path = os.path.join(OUT_DIR, filename)
        img = cv2.rotate(item.image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(out_path, img)
        print("Saved:", out_path)

print("✅ Export fertig. Bilder liegen in:", OUT_DIR)