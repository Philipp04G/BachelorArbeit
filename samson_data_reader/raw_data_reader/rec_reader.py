import os
import re
import yaml
import cv2
import numpy as np
from samson_data_reader.raw_data_reader.abc_raw_data_reader import RawDataReader
from samson_data_reader.datacalsses import *


class RecReader(RawDataReader):

    def __init__(self, path_rec=None, path_calib=None, image_size=None, load_images=True,
                 start_image_id=0, end_image_id=np.inf, **kwargs):

        self.path_rec = path_rec
        self.path_calib = path_calib
        self.load_images = load_images
        assert start_image_id < end_image_id
        self.start_image_id = start_image_id
        self.end_image_id = end_image_id

        if image_size is not None and not isinstance(image_size, np.ndarray):
            image_size = np.array(image_size)
        self.image_size = image_size

        # determine the camera name
        rec_file_name = os.path.basename(path_rec)
        camera_name_idx = re.search(r'SAMSON\d', rec_file_name)
        assert camera_name_idx is not None, 'Unable to find camera name within the rec file name'
        self.camera_name = rec_file_name[camera_name_idx.regs[0][0]: camera_name_idx.regs[0][1]]

        # some data about the image format and rectification which are written in __iter__
        self.rec_file = None
        self.rec_map1, self.rec_map2 = None, None
        self.raw_image_size = None
        self.raw_image_channel = None
        self.raw_image_conversion = None

        # some data about the camera which are written in __iter__
        self.camera_intrinsic = None
        self.calib_image_size = None
        self.stereo_baseline = None

        self._next_image = None

    def next_time(self):
        if self._next_image:
            return self._next_image.time
        else:
            return np.inf

    @staticmethod
    def read_calibration_file(path_calib):
        with open(path_calib, "r") as file:
            calib = yaml.safe_load(file)
            K = np.array(calib["cameraMatrix"])
            D = np.array(calib["distCoeffs"])
            if "rotation" in calib:
                R = np.array(calib["rotation"])
            else:
                R = np.eye(3, 3, dtype=np.float32)
            if "projectionMatrix" in calib:
                P = np.array(calib["projectionMatrix"])
                # calculate the stereo camera baseline
                baseline = -P[0, 3] / P[0, 0]
            else:
                P = K.copy()
                baseline = 0
            image_size = np.array(calib["imageSize"])[::-1]
            return K, D, P, R, image_size, baseline


    def read_rec_header(self, rec_file):

        file_header = rec_file.read(4 * 3)
        encoding, width, height = np.frombuffer(file_header, dtype=np.uint32)
        raw_image_size = np.array([height, width])
        raw_image_channel, raw_image_conversion = self.rec_encoding_to_cv_color(encoding)
        return raw_image_size, raw_image_channel, raw_image_conversion

    @staticmethod
    def rec_encoding_to_cv_color(encoding):
        """
        Get number of channels and cv2 color conversion for rec encoding.
        """

        # The encoding is equal to the pylon encoding of enum PixelFormatEnums
        if encoding == 32:  # bayerBG encoding
            channels = 1
            conv = cv2.COLOR_BayerBG2RGB
        elif encoding == 60:  # YCbCr422_8 encoding
            channels = 2
            conv = cv2.COLOR_YUV2RGB_YUY2
        elif encoding == 55:  # rgb
            channels = 3
            conv = None
        else:  # legacy .rec files
            raise AttributeError(f'unknown encoding {encoding}')

        return channels, conv

    def read_image(self, rec_file, load_image=True):

        header_data = rec_file.read(8 * 4)
        if len(header_data) == 0:
            return None

        # read the header information
        header = np.frombuffer(header_data, dtype=np.uint64)
        payload_bytes, image_id, secs, nsecs = header
        timestamp = secs + nsecs * 1e-9

        # read the image
        if load_image:
            data = rec_file.read(payload_bytes)
            image = np.frombuffer(data, dtype=np.uint8).reshape(*self.raw_image_size, self.raw_image_channel)
            if self.raw_image_conversion is not None:
                image = cv2.cvtColor(image, self.raw_image_conversion)

            # rectify the image
            if self.rec_map1 is not None:
                image = cv2.remap(image, self.rec_map1, self.rec_map2,
                                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # resize the image
            if self.image_size is not None:
                image = cv2.resize(image, self.image_size[::-1])
        else:
            rec_file.seek(payload_bytes, os.SEEK_CUR)
            image = None

        return ImageData(
            time=timestamp,
            image_id=image_id,
            image=image,
            camera_name=self.camera_name,
            camera_calibration=CameraIntrinsicData(
                camera_intrinsic=self.camera_intrinsic,
                camera_resolution=self.calib_image_size
            )
        )

    def __iter__(self):

        # read the rec file
        self.rec_file = open(self.path_rec, "rb")
        self.raw_image_size, self.raw_image_channel, self.raw_image_conversion = self.read_rec_header(self.rec_file)

        # read the calibration file
        if self.path_calib is not None:
            K, D, P, R, calib_image_size, stereo_baseline = self.read_calibration_file(self.path_calib)
            # remember camera attributes to write them into ImageData.CameraIntrinsicData
            self.camera_intrinsic = P[:3, :3]
            self.calib_image_size = calib_image_size
            self.stereo_baseline = stereo_baseline

            # create the calibration maps
            # check if the calibration must be scaled for image rectification
            # ATTENTION: The camera parameters that are returned by the dataset must NOT be scaled
            if not np.array_equal(self.raw_image_size, calib_image_size):
                scale_y, scale_x = self.raw_image_size / calib_image_size
                K_scaled = K.copy()
                P_scaled = P.copy()
                K_scaled[0, :] *= scale_x
                K_scaled[1, :] *= scale_y
                P_scaled[0, :] *= scale_x
                P_scaled[1, :] *= scale_y
                self.rec_map1, self.rec_map2 = cv2.initUndistortRectifyMap(K_scaled, D, R, P_scaled,
                                                                           self.raw_image_size[::-1], cv2.CV_32FC2)
            else:
                self.rec_map1, self.rec_map2 = cv2.initUndistortRectifyMap(K, D, R, P,
                                                                           self.raw_image_size[::-1], cv2.CV_32FC2)

        # read the first image so the next time can be told
        self._next_image = self.read_image(self.rec_file)

        return self


    def __next__(self):

        # check if the start image id is larger than the next image id. if yes skip images without exensive loading
        while self._next_image is not None and self._next_image.image_id < self.start_image_id:
            self._next_image = self.read_image(self.rec_file, load_image=self._next_image.image_id==self.start_image_id-1)
        # check if there is a next image
        if self._next_image is None:
            raise StopIteration
        # check if the next image id is larger than the end image id. If yes stop the iteration
        if self._next_image.image_id > self.end_image_id:
            raise StopIteration


        current_image = self._next_image
        self._next_image = self.read_image(self.rec_file, load_image=self.load_images)

        return current_image


class StereoRecReader(RawDataReader):

    def __init__(self, path_rec_left, path_rec_right, path_calib_left=None, path_calib_right=None, image_size=None, **kwargs):

        self.rec_reader_left = RecReader(path_rec=path_rec_left, path_calib=path_calib_left, image_size=image_size, **kwargs)
        self.rec_reader_right = RecReader(path_rec=path_rec_right, path_calib=path_calib_right, image_size=image_size, **kwargs)

        self.rec_gen_left, self.rec_gen_right = None, None

    def next_time(self):
        return self.rec_reader_left.next_time()

    def __iter__(self):
        self.rec_gen_left = iter(self.rec_reader_left)
        self.rec_gen_right = iter(self.rec_reader_right)
        return self

    def __next__(self):
        image_left = next(self.rec_gen_left)
        image_right = next(self.rec_gen_right)

        assert image_left.image_id == image_right.image_id

        if self.rec_gen_left.stereo_baseline is None:
            stereo_baseline = None
        else:
            stereo_baseline = max(self.rec_gen_left.stereo_baseline, self.rec_gen_right.stereo_baseline)

        return StereoImageData(
            time=image_left.time,
            image_data_left=image_left,
            image_data_right=image_right,
            stereo_baseline=stereo_baseline
        )







if __name__ == '__main__':

    _image_size = (768, 896)
    _data_dir = '/media/juri/T7/rawDatasets/SAMSON_IFAM_Basler/20240821_ApfelGroese/'
    _path_rec_left = os.path.join(_data_dir, '2024-08-21_10-17-12_A27_Groeßenmessung_vorher_SAMSON3_1724228232.rec')
    _path_rec_right = os.path.join(_data_dir, '2024-08-21_10-17-12_A27_Groeßenmessung_vorher_SAMSON4_1724228232.rec')
    _path_calib_left = None  # os.path.join(_data_dir, 'SAMSON3_SAMSON4_stereo.yaml')
    _path_calib_right = None  #  os.path.join(_data_dir, 'SAMSON4_SAMSON3_stereo.yaml')

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    _data_reader = RecReader(path_rec=_path_rec_left, path_calib=_path_calib_left, image_size=None,
                             start_image_id=2500, end_image_id=2600)
    for _data in _data_reader:
        print(f'image id {_data.image_id}')
        cv2.imshow(f'image', cv2.cvtColor(_data.image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)

    #_data_reader = StereoRecReader(path_rec_left=_path_rec_left, path_rec_right=_path_rec_right,
    #                               path_calib_left=_path_calib_left, path_calib_right=_path_calib_right,
    #                               image_size=_image_size)
    #_last_time = 0
    #for _data in _data_reader:
    #    #cv2.imshow(f'image left', _data.image_left)
    #    #cv2.imshow(f'image right', _data.image_right)
    #    #cv2.waitKey(1)
    #    print(f'time: {_data.time - _last_time}')
    #    _last_time = _data.time

