import numpy as np
import cv2
from samson_data_reader.data_reader import DataReader
from camera_calibration.calibrate_camera import CharucoCameraCalibrator


if __name__ == '__main__':

    #path_data_dir = '/media/juri/T7/rawDatasets/baslerStereo/20250818_calib'
    path_data_dir = '/home/juri/Desktop/tmp_calib'

    data_reader = DataReader(path_data_dir=path_data_dir, stereo_images=True, rectify_images=True,
                                  start_image_id=0, end_image_id=np.inf,
                                  )

    camera_calibrator = CharucoCameraCalibrator(path_data_dir=path_data_dir, calibration_board_size='coarse')

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    for data in data_reader:

        debug_frame_left, charuco_dict_left = camera_calibrator.detect_charuco_markers(data.image_data_left.image)
        debug_frame_right, charuco_dict_right = camera_calibrator.detect_charuco_markers(data.image_data_right.image)

        # select the intersecting charuco points for the stereo calibration
        inter_ids, inter_indices_left, inter_indices_right = np.intersect1d(charuco_dict_left['ids'],
                                                                            charuco_dict_right['ids'],
                                                                            return_indices=True)
        object_points = charuco_dict_left['objectPoints'][inter_indices_left]
        corners_left = charuco_dict_left['corners'][inter_indices_left]
        corners_right = charuco_dict_right['corners'][inter_indices_right]

        corner_diff = corners_left - corners_right
        disparity = corner_diff[..., 0:1]

        # TODO: calculate the 3D coordinates of each charoucu corner
        camera_intrinsic = data.image_data_left.camera_calibration.camera_intrinsic
        depth = camera_intrinsic[0, 0] / disparity * data.stereo_baseline
        corner_left_pad = np.pad(corners_left, ((0, 0), (0, 1)), constant_values=1)
        corner_rwc = depth * (np.linalg.inv(camera_intrinsic)[np.newaxis] @ corner_left_pad[:, :, np.newaxis])[:, :, 0]

        # measure the distance error of nearby corners in 3D
        corner_distances = []
        board_size = camera_calibrator.board.getChessboardSize()
        for y in range(board_size[1] - 2):
            for x in range(board_size[0] - 2):
                marker1_id = (board_size[0] - 1) * y + x
                marker2_id = (board_size[0] - 1) * y + (x+1)
                marker3_id = (board_size[0] - 1) * (y+1) + x
                marker1_idx = np.where(inter_ids==marker1_id)[0]
                marker2_idx = np.where(inter_ids == marker2_id)[0]
                marker3_idx = np.where(inter_ids == marker3_id)[0]
                if len(marker1_idx) == 1 and len(marker2_idx) == 1:
                    corner_distances.append(np.linalg.norm(corner_rwc[marker1_idx[0]] - corner_rwc[marker2_idx[0]]))
                if len(marker1_idx) == 1 and len(marker3_idx) == 1:
                    corner_distances.append(np.linalg.norm(corner_rwc[marker1_idx[0]] - corner_rwc[marker3_idx[0]]))
        corner_distances = np.array(corner_distances)
        corner_error = np.abs(corner_distances -camera_calibrator.board.getSquareLength())

        print(f'Image {data.image_data_left.image_id}')
        print(f'vertical disparity: mean={np.mean(np.abs(corner_diff[..., 1])):.3f}, max={np.max(np.abs(corner_diff[..., 1])):.3f}')
        print(f'3D corner error: mean={np.mean(corner_error)*1e3:.2f}mm, max={np.max(corner_error)*1e3:.2f}mm')

        debug_frame = np.hstack([debug_frame_left, debug_frame_right])
        cv2.imshow("frame", cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0)