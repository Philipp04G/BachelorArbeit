import os
import cv2
from samson_data_reader.data_reader import DataReader
from samson_data_reader.datacalsses import ImageData

# -----------------------------------------------------
# SETTINGS
# -----------------------------------------------------

DATA_DIR = "D:\Studium\BACHELOR ARBEIT\samson_data_reader"   # Ordner, wo die REC-Dateien liegen
OUT_DIR = os.path.join(DATA_DIR, "exported_dataset_test")


EXPORT_EVERY = 3            # jedes 10. Bild
TRAIN_RATIO = 0.8            # 80% Training, 20% Validation

# Create YOLO folder structure
TRAIN_IMG = os.path.join(OUT_DIR, "images", "train")
VAL_IMG   = os.path.join(OUT_DIR, "images", "val")
TRAIN_LBL = os.path.join(OUT_DIR, "labels", "train")
VAL_LBL   = os.path.join(OUT_DIR, "labels", "val")

for d in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
    os.makedirs(d, exist_ok=True)


# -----------------------------------------------------
# MAIN: READ STEREO IMAGES
# -----------------------------------------------------

dr = DataReader(
    path_data_dir=DATA_DIR,
    stereo_images=False,
    load_dvso=False,
    load_gps=False,
    load_lidar=False,
    image_size=None
)

counter = 0
train_counter = 0
val_counter = 0
img_counter = 1
for item in dr:

    if isinstance(item, ImageData):

        # Only export every Nth image
        if counter % EXPORT_EVERY != 0:
            counter += 1
            continue

    filename = f"{int(img_counter):06d}.jpg"  # z.B. 000001.jpg

    # Train/val split
    total = train_counter + val_counter
    if total == 0:
        portion = 0
    else:
        portion = train_counter / total

    if portion < TRAIN_RATIO:
        target_dir = TRAIN_IMG
        train_counter += 1
    else:
        target_dir = VAL_IMG
        val_counter += 1

    out_path = os.path.join(OUT_DIR, target_dir, filename)
    img = cv2.rotate(item.image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(out_path, img)

    print(f"Exported ")
    img_counter += 1
    counter += 1

print("\n------------------------------------------------")
print("✅ Export vollständig abgeschlossen!")
print("→ Train Bilder:", train_counter)
print("→ Val Bilder:  ", val_counter)
print("→ Ausgabeordner:", OUT_DIR)
print("------------------------------------------------")