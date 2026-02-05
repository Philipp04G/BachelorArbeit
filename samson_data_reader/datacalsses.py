from dataclasses import dataclass
import numpy as np


@dataclass
class GpsData:
    time: float
    position: np.ndarray
    position_variance: np.ndarray


@dataclass
class LidarPointsData:
    time: float
    points: np.ndarray
    points_time: np.ndarray


@dataclass
class VisualOdometryData:
    time: float
    image_id: int
    motion: np.ndarray
    object_detection: list


@dataclass
class DeepVisualStereoOdometryData:
    image_depth: np.ndarray
    image_depth_var: np.ndarray
    lidar_depth: np.ndarray = None
    optical_flow: np.ndarray = None
    optical_flow_var: np.ndarray = None
    visual_odometry: np.ndarray = None
    vo_rotation_var: np.ndarray = None
    vo_translation_var: np.ndarray = None
    trajectory: np.ndarray = None
    trajectory_rotation_var: np.ndarray = None
    trajectory_translation_var: np.ndarray = None


@dataclass
class DetectionData:
    bbox: list
    segmentation: list = None


@dataclass
class CameraIntrinsicData:
    camera_intrinsic: np.ndarray
    camera_resolution: np.ndarray


@dataclass
class ImageData:
    time: float
    image_id: int
    image: np.ndarray
    camera_name: str
    camera_calibration: CameraIntrinsicData
    dvso_data: DeepVisualStereoOdometryData = None
    detection_data: DetectionData = None


@dataclass
class StereoImageData:
    time: float
    image_data_left: ImageData
    image_data_right: ImageData
    stereo_baseline: float
