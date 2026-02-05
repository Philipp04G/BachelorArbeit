import os.path
import sys
import yaml
import cv2
import numpy as np

import camera_calibration.rec_utils
from samson_data_reader.data_reader import DataReader
from samson_data_reader.datacalsses import *


class CharucoCameraCalibrator:

    def __init__(self, path_data_dir, calibration_board_size='coarse', start_image_id=0, end_image_id=np.inf):

        self.data_reader = DataReader(path_data_dir=path_data_dir, stereo_images=True, rectify_images=False,
                                      start_image_id=start_image_id, end_image_id=end_image_id,
                                      )

        self.calib_flags = cv2.CALIB_FIX_INTRINSIC

        # create charuco board
        assert calibration_board_size in ['coarse', 'fine']
        if calibration_board_size == 'coarse':
            self.board = cv2.aruco.CharucoBoard([12, 9], 0.06, 0.045,
                                                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
        else:
            self.board = cv2.aruco.CharucoBoard([24, 17], 0.03, 0.022,
                                                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000))
        self.board.setLegacyPattern(True) # TODO: is this still required?
        self.detector = cv2.aruco.CharucoDetector(self.board)

        self.image_size = None
        self.cam_left_name = None
        self.cam_right_name = None


    def detect_charuco_markers(self, image: np.ndarray):

        debug_frame = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) < 10:
            if charuco_ids is None:
                print("no charuco ids")
            elif len(charuco_ids) < 10:
                print("too few markers detected", len(charuco_ids))
            charuco_corners = np.zeros((0, 1, 2))
            charuco_ids = np.zeros((0, 1))
            object_points = np.zeros((0, 1, 3))
        else:
            object_points, image_points = self.board.matchImagePoints(charuco_corners, charuco_ids)
            cv2.aruco.drawDetectedCornersCharuco(debug_frame, charuco_corners, charuco_ids, (255, 0, 0))

        charuco_dict = {
            'corners': np.squeeze(charuco_corners),
            'ids': np.squeeze(charuco_ids),
            'objectPoints': np.squeeze(object_points),
        }

        return debug_frame, charuco_dict

    def calibrate_mono(self, charuco_dict):

        object_points = [x for x in charuco_dict['objectPoints'] if x.shape[0] > 0]
        corners = [x for x in charuco_dict['corners'] if x.shape[0] > 0]
        nretval, camera_matrix, dist_coeffs, nrvecs, ntvecs = \
            cv2.calibrateCamera(object_points, corners, self.image_size, None, None, flags=0)

        calib_dict = {
            'cameraMatrix': camera_matrix,
            'distCoeffs': dist_coeffs,
            'imageSize': self.image_size
        }

        return calib_dict

    @staticmethod
    def undistort(img, K, D, R=None, P=None):
        if P is None:
            P = K.copy()
        if R is None:
            R = np.eye(3, 3, dtype=np.float32)
        image_size = (img.shape[1], img.shape[0])
        map1, map2 = cv2.initUndistortRectifyMap(K, D, R, P, image_size, cv2.CV_16SC2)
        undistorted_img = cv2.remap(
            img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        return undistorted_img

    @staticmethod
    def writeCalib(filename, image_size, K, D, R, P):
        with open(filename, "w") as file:
            yaml.safe_dump(
                {
                    "imageSize": image_size,
                    "cameraMatrix": K.tolist(),
                    "distCoeffs": D.tolist(),
                    "rotation": R.tolist(),
                    "projectionMatrix": P.tolist(),
                },
                file,
                default_flow_style=None,
            )
        print("wrote:", filename)

    def calibrate(self):

        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

        stereo_charuco_dicts = None
        stereo_images = None
        left_count, right_count = 0, 0
        charuco_dict_left  = {'corners': [], 'ids': [], 'objectPoints': [], 'imageSize': None}
        charuco_dict_right = {'corners': [], 'ids': [], 'objectPoints': [], 'imageSize': None}
        for data in self.data_reader:

            if self.image_size is None:
                self.image_size = data.image_data_left.image.shape[:2][::-1]
            if self.cam_left_name is None:
                self.cam_left_name = data.image_data_left.camera_name
            if self.cam_right_name is None:
                self.cam_right_name = data.image_data_right.camera_name


            print(f'Image {data.image_data_left.image_id}:')

            debug_frame_left, tmp_charuco_dict_left = self.detect_charuco_markers(data.image_data_left.image)
            debug_frame_right, tmp_charuco_dict_right = self.detect_charuco_markers(data.image_data_right.image)

            debug_frame = np.hstack([debug_frame_left, debug_frame_right])
            cv2.imshow("frame", cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(0)

            add_left, add_right, add_stereo = False, False, False
            if key & 0xFF == ord("w"):
                add_left = True
                add_right = True
            elif key & 0xFF == ord("a"):
                add_left = True
            elif key & 0xFF == ord("d"):
                add_right = True
            elif key & 0xFF == ord("s"):
                add_left = True
                add_right = True
                add_stereo = True
            elif key & 0xFF == ord("b"):
                break
            elif key & 0xFF == ord("q"):
                sys.exit(0)

            if add_left:
                n_points_left = len(tmp_charuco_dict_left['ids'])
                left_count += n_points_left
                for key, value in tmp_charuco_dict_left.items():
                    charuco_dict_left[key].append(value)
            else:
                n_points_left = 0
            if add_right:
                n_points_right = len(tmp_charuco_dict_right['ids'])
                right_count += n_points_right
                for key, value in tmp_charuco_dict_right.items():
                    charuco_dict_right[key].append(value)
            else:
                n_points_right = 0
            if add_stereo:
                stereo_charuco_dicts = (tmp_charuco_dict_left, tmp_charuco_dict_right)
                stereo_images = (data.image_data_left.image, data.image_data_right.image)
                print(f'set stereo calibration image to: {data.image_data_left.image_id}')
            print(f'adding left {n_points_left}, total: {left_count}')
            print(f'adding right {n_points_right}, total: {right_count}')

        print('############## Calibrate mono cameras ##################')
        calib_dict_left = self.calibrate_mono(charuco_dict_left)
        calib_dict_right = self.calibrate_mono(charuco_dict_right)

        print('############## Calibrate stereo camera ##################')
        # select the intersecting charuco points for the stereo calibration
        inter_ids, inter_indices_left, inter_indices_right = np.intersect1d(stereo_charuco_dicts[0]['ids'],
                                                                            stereo_charuco_dicts[1]['ids'],
                                                                            return_indices=True)
        inter_object_points = stereo_charuco_dicts[0]['objectPoints'][inter_indices_left][np.newaxis]
        inter_corners_left = stereo_charuco_dicts[0]['corners'][inter_indices_left][np.newaxis]
        inter_corners_right = stereo_charuco_dicts[1]['corners'][inter_indices_right][np.newaxis]

        ret_stereo_calibrate = cv2.stereoCalibrate(
            inter_object_points,
            inter_corners_left,
            inter_corners_right,
            calib_dict_left['cameraMatrix'],
            calib_dict_left['distCoeffs'],
            calib_dict_right['cameraMatrix'],
            calib_dict_right['distCoeffs'],
            self.image_size,
            flags=self.calib_flags,
        )
        ret_sc, _, _, _, _, R, T, E, F = ret_stereo_calibrate

        ret_stereo_rect = cv2.stereoRectify(
            calib_dict_left['cameraMatrix'],
            calib_dict_left['distCoeffs'],
            calib_dict_right['cameraMatrix'],
            calib_dict_right['distCoeffs'],
            self.image_size,
            R, T, flags=cv2.CALIB_ZERO_DISPARITY
        )
        R_l, R_r, P_l, P_r, Q, roi_L, roi_R  = ret_stereo_rect

        ################## save the camera calibration  ###############################
        save_path_left = os.path.join(self.data_reader.path_data_dir, f'{self.cam_left_name}_{self.cam_right_name}_stereo.yaml')
        self.writeCalib(save_path_left, self.image_size, calib_dict_left['cameraMatrix'], calib_dict_left['distCoeffs'], R_l, P_l)

        save_path_left = os.path.join(self.data_reader.path_data_dir, f'{self.cam_right_name}_{self.cam_left_name}_stereo.yaml')
        self.writeCalib(save_path_left, self.image_size, calib_dict_right['cameraMatrix'], calib_dict_right['distCoeffs'], R_r, P_r)

        # ################## plot the rectified stereo images  ###############################
        image_left_rect = self.undistort(stereo_images[0], calib_dict_left['cameraMatrix'], calib_dict_left['distCoeffs'],
                                         R_l, P_l)
        image_right_rect = self.undistort(stereo_images[1], calib_dict_right['cameraMatrix'], calib_dict_right['distCoeffs'],
                                          R_r, P_r)
        image_left_rect = cv2.cvtColor(image_left_rect, cv2.COLOR_RGB2BGR)
        image_right_rect = cv2.cvtColor(image_right_rect, cv2.COLOR_RGB2BGR)

        def mouse_cb(event, x, y, flags, param):
            tmp_L = image_left_rect.copy()
            tmp_R = image_right_rect.copy()
            h, w = tmp_L.shape[:2]
            _, _, window_left_w, window_left_h = cv2.getWindowImageRect('image_left_rect')
            _, _, window_right_w, window_right_h = cv2.getWindowImageRect('image_right_rect')
            cv2.line(tmp_L, (0, y), (w, y), (0, 255, 0), int(h/window_left_h))
            cv2.line(tmp_R, (0, y), (w, y), (0, 255, 0), int(h/window_right_h))
            cv2.imshow('image_left_rect', tmp_L)
            cv2.imshow('image_right_rect', tmp_R)

        cv2.namedWindow('image_left_rect', cv2.WINDOW_NORMAL)
        cv2.namedWindow('image_right_rect', cv2.WINDOW_NORMAL)
        cv2.imshow('image_left_rect', image_left_rect)
        cv2.imshow('image_right_rect', image_right_rect)
        cv2.moveWindow('image_right_rect', 680, 0)
        cv2.setMouseCallback('image_left_rect', mouse_cb)
        cv2.setMouseCallback('image_right_rect', mouse_cb)
        cv2.waitKey(0)

        print(':)')


if __name__ == '__main__':

    _path_data_dir = '/media/juri/T7/rawDatasets/SAMSON_IFAM_Basler/20240821_ApfelGroese/calib'

    camera_calibrator = CharucoCameraCalibrator(path_data_dir=_path_data_dir, calibration_board_size='fine',
                                                start_image_id=7728)
    camera_calibrator.calibrate()



    print(':)')