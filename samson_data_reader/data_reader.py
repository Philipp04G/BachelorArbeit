import glob
import os.path
import platform
import re
import numpy as np
from samson_data_reader.raw_data_reader.rec_reader import RecReader, StereoRecReader
# try to load ros bag reader. Try & except, so the system can run without ros
try:
    from samson_data_reader.raw_data_reader.rosbag_reader import RosBagReader
except ModuleNotFoundError:
    print('WARNING: Unable to import RosBagReader.')
from samson_data_reader.processed_data_reader.dvso_reader import DeepVisualStereoOdometryReader
from samson_data_reader.datacalsses import *

class DataReader:

    def __init__(self, path_data_dir, image_size=None,
                 rectify_images=True, path_camera_calib_left=None, path_camera_calib_right=None,
                 stereo_images=False, load_gps=False, load_lidar=False, load_dvso=False, load_object_detection=False,
                 dvso_dir_name=None, **kwargs):

        if stereo_images:
            assert not load_dvso and not load_object_detection, 'dvso and detection data is not supported with stereo images'

        print(stereo_images)
        self.path_data_dir = path_data_dir
        self.image_size = image_size
        self.dvso_name = dvso_dir_name
        self.kwargs = kwargs  # kwargs are added to all readers

        # which data should be returned
        self.stereo_images = stereo_images
        self.load_gps = load_gps
        self.load_lidar = load_lidar
        self.load_dvso = load_dvso
        self.load_detection_data = load_object_detection

        self.image_reader, ros_bag_reader = None, None
        self.raw_data_reader, self.processed_data_reader = None, None
        self.raw_data_generator = None

        if not rectify_images:
            self.path_camera_calib_left = None
            self.path_camera_calib_right = None
        elif path_camera_calib_left is None and path_camera_calib_right is None:
            self.path_camera_calib_left, self.path_camera_calib_right = self.search_camera_calib()
        else:
            self.path_camera_calib_left = path_camera_calib_left
            self.path_camera_calib_right = path_camera_calib_right

        # some path to data dirs. They are se when loading data
        self.path_dvso_dir = None


    def search_camera_calib(self):

        path_calib_left, path_calib_right = None, None

        # check the data folder for calibration files
        calib_file_list = []
        for name in os.listdir(self.path_data_dir):
            if re.match(r'SAMSON\d_SAMSON\d_stereo.yaml', name):
                calib_file_list.append(name)
        if len(calib_file_list) == 2:
            calib_file_list.sort()
            path_calib_left = os.path.join(self.path_data_dir, calib_file_list[0])
            path_calib_right = os.path.join(self.path_data_dir, calib_file_list[1])


        # Check out the default path for different hosts
        if path_calib_left is None or path_calib_right is None:
            if platform.node() == 'juri-ThinkPad':
                calib_dir = '/home/juri/SAMSON/cameraCalibration'
                path_calib_left = os.path.join(calib_dir, 'SAMSON3_SAMSON4_stereo.yaml')
                path_calib_right = os.path.join(calib_dir, 'SAMSON4_SAMSON3_stereo.yaml')
            # jetson orin from HAW
            elif platform.node() == 'ubuntu':
                calib_dir = '/media/data/Samson_Data/cameraCalibration'
                path_calib_left = os.path.join(calib_dir, 'SAMSON1_SAMSON2_stereo.yaml')
                path_calib_right = os.path.join(calib_dir, 'SAMSON2_SAMSON1_stereo.yaml')
            else:
                raise AttributeError('Unknown platform')

        assert path_calib_left is not None and path_calib_right is not None, 'Could not find camera calibration file'

        return path_calib_left, path_calib_right

    def create_raw_data_reader(self):

        data_reader_list = []

        # search for the image files
        path_rec_files = glob.glob(os.path.join(self.path_data_dir, '*.rec'))
        #assert len(path_rec_files) == 2, 'There should be two rec files'
        # sort the list, so that the camera with the smaller number is first
        path_rec_files.sort()
        if self.stereo_images:
            self.image_reader = StereoRecReader(
                print(tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt),
                path_rec_left=path_rec_files[0],
                path_rec_right=path_rec_files[1],
                path_calib_left=self.path_camera_calib_left,
                path_calib_right=self.path_camera_calib_right,
                image_size=self.image_size,
                **self.kwargs
            )
        else:
            self.image_reader = RecReader(
                path_rec=path_rec_files[0],
                path_calib=self.path_camera_calib_left,
                image_size=self.image_size,
                **self.kwargs
            )
        data_reader_list.append(self.image_reader)

        # search for bag file
        if self.load_gps or self.load_lidar:
            path_rosbags = glob.glob(os.path.join(self.path_data_dir, '*.bag'))
            if len(path_rosbags) > 1:
                # remove .bag with "no_lidar" in name
                path_rosbags = [p for p in path_rosbags if 'no_lidar' not in os.path.basename(p)]
            assert len(path_rosbags) == 1
            # create the rosbag data reader
            self.ros_bag_reader = RosBagReader(
                path_rosbag=path_rosbags[0],
                read_gps=self.load_gps,
                read_lidar=self.load_lidar,
                **self.kwargs
            )

            data_reader_list.append(self.ros_bag_reader)


        return data_reader_list

    def create_processed_data_reader(self):

        data_reader_list = []

        # create data reader for dvso data
        if self.load_dvso:
            if self.dvso_name:
                path_dvso_dir = os.path.join(self.path_data_dir, self.dvso_name)
                assert os.path.exists(self.path_data_dir)
            else:
                # search for the dvso dir
                dvso_dirs = []
                for name in os.listdir(self.path_data_dir):
                    if os.path.isdir(os.path.join(self.path_data_dir, name)) and re.match(r'dvs?o_*', name):
                        dvso_dirs.append(name)
                assert len(dvso_dirs) > 0, 'no deep visual stereo odometry predictions could be found'
                assert len(dvso_dirs) == 1, 'multiple deep visual stereo odometry predictions where found:' + ', '.join(dvso_dirs)
                path_dvso_dir = os.path.join(self.path_data_dir, dvso_dirs[0])

            data_reader_list.append(
                DeepVisualStereoOdometryReader(path_data_dir=path_dvso_dir, **self.kwargs)
            )
            self.path_dvso_dir = path_dvso_dir

        if self.load_detection_data:
            raise NotImplemented

        return data_reader_list

    def __iter__(self):

        # create data reader and generator for the raw data
        self.raw_data_reader = self.create_raw_data_reader()
        self.raw_data_generator = [iter(dr) for dr in self.raw_data_reader]

        # create data reader for the processed data
        self.processed_data_reader = self.create_processed_data_reader()

        return self


    def __next__(self):

        # TODO: check out which raw data reader has the closest timestamp
        next_raw_reader_idx = np.argmin([r.next_time() for r in self.raw_data_reader])

        # get the next reader data
        data = next(self.raw_data_reader[next_raw_reader_idx])

        # if data is image data, add all available already processed data to the image
        if isinstance(data, ImageData):
            for pdr in self.processed_data_reader:
                pdr.add_data(data)

        return data





if __name__ == '__main__':

    import cv2

    ai_name = 'un26w2en_step500000'

    _path_data_dir = '/media/juri/T7/rawDatasets/SAMSON_IFAM_Basler/2025-07-02_08-47-04_Samson_Technologietag_Wellant_Sensorbox1'
    #_path_data_dir = '/media/juri/T7/rawDatasets/SAMSON_IFAM_Basler/2024-08-08_16-49-45_Fruchtentwicklung_D19_Targets_Esteburg_Sensorbox1'
    _data_reader = DataReader(path_data_dir=_path_data_dir,
                             stereo_images=False,
                             load_gps=False,
                             load_lidar=False,
                             load_dvso=False,
                             load_object_detection=False,
                             #start_image_id=9000, #12298
                             )

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    for _data in _data_reader:
        print(f'image id {_data.image_id}')
        cv2.imshow(f'image', cv2.rotate(cv2.cvtColor(_data.image, cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE))
        cv2.waitKey(1)