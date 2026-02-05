from samson_data_reader.data_reader import DataReader
from samson_data_reader.datacalsses import ImageData, GpsData, LidarPointsData

data_dir = "D:\Studium\BACHELOR ARBEIT\samson_data_reader"

# explizit die beiden Stereo-YAMLs angeben (Links/Rechts):
path_calib_left  = f"{data_dir}/SAMSON3_SAMSON4_stereo.yaml"   # <- deinen Dateinamen einsetzen
path_calib_right = f"{data_dir}/SAMSON4_SAMSON3_stereo.yaml"   # <- die zweite Datei

dr = DataReader(
    path_data_dir=data_dir,
    stereo_images=False,              # Mono-Iteration (Bilder & Messungen in Zeitreihenfolge)
    load_gps=True, load_lidar=True,   # *.bag wird eingemischt
    load_dvso=True,                   # DVSO-Outputs anhängen (Depth/Flow/Pose/…)
    dvso_dir_name="dvso_6zhrpwd5_step560000",
    path_camera_calib_left=path_calib_left,
    path_camera_calib_right=path_calib_right,
    image_size=(768, 896),            # optionales Resize (H,W)
)

for item in dr:
    if isinstance(item, ImageData):
        has_depth = bool(item.dvso_data and item.dvso_data.image_depth is not None)
        has_pose  = bool(item.dvso_data and item.dvso_data.trajectory is not None)
        print(f"IMG {item.image_id} | K shape={item.camera_calibration.camera_intrinsic.shape} | depth={has_depth} | pose={has_pose}")
    elif isinstance(item, GpsData):
        print("GPS", item.position)   # lokale Meter (vereinfachte Umrechnung)
    elif isinstance(item, LidarPointsData):
        print("LIDAR", item.points.shape)
