import os
import glob
import numpy as np
from samson_data_reader.datacalsses import *
from samson_data_reader.processed_data_reader.processed_data_reader import ProcessedDataReader


class DeepVisualStereoOdometryReader(ProcessedDataReader):

    def __init__(self, path_data_dir, load_dvso_image_depth=True, load_dvso_lidar_depth=False, load_dvso_optical_flow=True, **kwargs):

        self.path_data_dir = path_data_dir
        self.load_image_depth = load_dvso_image_depth
        self.load_lidar_depth = load_dvso_lidar_depth
        self.load_optical_flow = load_dvso_optical_flow

        # preload odometry data and trajectory data
        path_odometry = os.path.join(self.path_data_dir, f'odometry.txt')
        path_odometry_var = os.path.join(self.path_data_dir, f'odometry_var.txt')
        path_trajectory = os.path.join(self.path_data_dir, f'trajectory.txt')
        path_trajectory_var = os.path.join(self.path_data_dir, f'trajectory_var.txt')

        if os.path.exists(path_odometry):
            self.odometry_dict = self.read_odometry_file(path_odometry)
        else:
            self.odometry_dict = {}
        if os.path.exists(path_odometry_var):
            self.vo_rotation_var_dict, self.vo_translation_var_dict = self.read_odometry_var_file(path_odometry_var)
        else:
            self.vo_rotation_var_dict, self.vo_translation_var_dict = {}, {}
        if os.path.exists(path_trajectory):
            self.trajectory_dict = self.read_odometry_file(path_trajectory)
        else:
            self.trajectory_dict = {}
        if os.path.exists(path_trajectory_var):
            self.trajectory_rotation_var_dict, self.trajectory_translation_var_dict = self.read_odometry_var_file(path_trajectory_var)
        else:
            self.trajectory_rotation_var_dict, self.trajectory_translation_var_dict = {}, {}

    @staticmethod
    def read_odometry_file(file_path):

        with open(file_path, 'r') as file:
            lines = file.readlines()

        odometry_dict = {}
        for line in lines:
            img_id, odometry = line.split(':')
            odometry = np.reshape(np.fromstring(odometry, dtype=np.float32, sep=' '), (4, 4))
            odometry_dict[int(img_id)] = odometry

        return odometry_dict

    @staticmethod
    def read_odometry_var_file(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        translation_var_dict = {}
        rotation_var_dict = {}
        for line in lines:
            img_id, odometry_var = line.split(':')
            odometry_var = np.fromstring(odometry_var, dtype=np.float32, sep=' ')
            rotation_var_dict[int(img_id)] = odometry_var[:3]
            translation_var_dict[int(img_id)] = odometry_var[3:]
        return rotation_var_dict, translation_var_dict


    def add_data(self, image_data: ImageData):


        camera_name = image_data.camera_name
        path_depth_dir = os.path.join(self.path_data_dir, f'{camera_name}_depth')
        path_depth_var_dir = os.path.join(self.path_data_dir, f'{camera_name}_depth_var')
        path_flow_dir = os.path.join(self.path_data_dir, f'{camera_name}_flow')
        path_flow_var_dir = os.path.join(self.path_data_dir, f'{camera_name}_flow_var')
        path_cam_lidar_fusion = os.path.join(self.path_data_dir, f'{camera_name}_cam_lidar_fusion')

        image_id = image_data.image_id
        if self.load_image_depth:
            image_depth = self.find_numpy_data(path_depth_dir, image_id)
            image_depth_var = self.find_numpy_data(path_depth_var_dir, image_id)
        else:
            image_depth = None
            image_depth_var = None

        if self.load_lidar_depth:
            lidar_depth = self.find_numpy_data(path_cam_lidar_fusion, image_id)
        else:
            lidar_depth = None

        if self.load_optical_flow:
            optical_flow = self.find_numpy_data(path_flow_dir, image_id)
            optical_flow_var = self.find_numpy_data(path_flow_var_dir, image_id)
        else:
            optical_flow = None
            optical_flow_var = None

        dvso_data = DeepVisualStereoOdometryData(
            image_depth=image_depth,
            image_depth_var=image_depth_var,
            lidar_depth=lidar_depth,
            optical_flow=optical_flow,
            optical_flow_var=optical_flow_var,
            visual_odometry=self.odometry_dict.get(image_id),
            vo_rotation_var=self.vo_rotation_var_dict.get(image_id),
            vo_translation_var=self.vo_translation_var_dict.get(image_id),
            trajectory=self.trajectory_dict.get(image_id),
            trajectory_rotation_var=self.trajectory_rotation_var_dict.get(image_id),
            trajectory_translation_var=self.trajectory_translation_var_dict.get(image_id)
        )

        image_data.dvso_data = dvso_data
        return image_data

    @staticmethod
    def find_numpy_data(path_dir, image_id):

        file_list = glob.glob(os.path.join(path_dir, f'*{image_id}*.npy'))

        if len(file_list) == 0:
            return None
        elif len(file_list) == 1:
            return np.load(file_list[0])
        else:
            raise FileNotFoundError('Multiple files found: ' + ', '.join([os.path.basename(f) for f in file_list]))








if __name__ == '__main__':

    from samson_data_reader.raw_data_reader.rec_reader import RecReader

    _image_size = (768, 896)
    _data_dir = '/media/juri/T7/rawDatasets/SAMSON_IFAM_Basler/20240821_ApfelGroese/'
    _path_rec_left = os.path.join(_data_dir, '2024-08-21_10-17-12_A27_Groeßenmessung_vorher_SAMSON3_1724228232.rec')
    _path_calib_left = os.path.join(_data_dir, 'SAMSON3_SAMSON4_stereo.yaml')
    _data_reader = RecReader(path_rec=_path_rec_left, path_calib=_path_calib_left, image_size=_image_size)

    _path_dvso_data = os.path.join(_data_dir, '20240821_apple_size/dvo_75fsxg6c_step480000')
    dvso_reader = DeepVisualStereoOdometryReader(_path_dvso_data)

    for _data in _data_reader:
        dvso_reader.add_data(_data)

        print(':)')