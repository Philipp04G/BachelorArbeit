import tensorflow as tf
from samson_data_reader.data_reader import DataReader
from samson_data_reader.datacalsses import *


default_image_pre_proc = lambda x: tf.cast(x, tf.float32) / 255.

class DataReaderForDVSO:

    def __init__(self, path_data_dir, batch_size=1, image_size=None, image_pre_proc=default_image_pre_proc, prefetch=10,
                 path_camera_calib_left=None, path_camera_calib_right=None, **kwargs):

        self.batch_size = batch_size
        self.image_pre_proc = image_pre_proc
        self.prefetch = prefetch

        self.data_reader = DataReader(path_data_dir=path_data_dir, image_size=image_size,
                                      path_camera_calib_left=path_camera_calib_left,
                                      path_camera_calib_right=path_camera_calib_right,
                                      stereo_images=True,
                                      load_gps=False, load_lidar=False, load_dvso=False, load_object_detection=False,
                                      **kwargs)
        # iterate the data reader to load information about the data
        iter(self.data_reader)

    @property
    def camera_names(self):
        return [self.data_reader.image_reader.rec_reader_left.camera_name,
                self.data_reader.image_reader.rec_reader_right.camera_name]


    def _tf_data_gen(self):

        data_generator = iter(self.data_reader)

        try:
            current_image_data: StereoImageData = next(data_generator)
        except StopIteration:
            return

        while True:

            try:
                next_image_data: StereoImageData = next(data_generator)
            except StopIteration:
                return

            # prepare data
            yield (self.image_pre_proc(current_image_data.image_data_left.image),
                   self.image_pre_proc(next_image_data.image_data_left.image),
                   self.image_pre_proc(current_image_data.image_data_right.image),
                   current_image_data.image_data_left.camera_calibration.camera_resolution,
                   current_image_data.image_data_left.camera_calibration.camera_intrinsic,
                   np.array([current_image_data.stereo_baseline] + [0.]*5),
                   current_image_data.image_data_left.image_id)

            current_image_data = next_image_data


    def _get_tf_dataset_signature(self):

        image_shape = (*self.data_reader.image_size, 3)

        output_signature = [tf.TensorSpec(shape=image_shape, dtype=tf.float32, name='image_cam0t0'),
                            tf.TensorSpec(shape=image_shape, dtype=tf.float32, name='image_cam0t1'),
                            tf.TensorSpec(shape=image_shape, dtype=tf.float32, name='image_cam1t0'),
                            tf.TensorSpec(shape=(2,), dtype=tf.float32, name='cam_resolution'),
                            tf.TensorSpec(shape=(3, 3), dtype=tf.float32, name='cam_intrinsic'),
                            tf.TensorSpec(shape=(6,), dtype=tf.float32, name='cam_distance'),
                            tf.TensorSpec(shape=(), dtype=tf.int32, name='frame_id')]

        return output_signature

    def to_tf_dataset(self):

        output_signature = self._get_tf_dataset_signature()
        dataset = tf.data.Dataset.from_generator(self._tf_data_gen, output_signature=tuple(output_signature))


        if self.batch_size:
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
        if self.prefetch:
            dataset = dataset.prefetch(self.prefetch)

        return dataset



class DataReaderForStereoDepth(DataReaderForDVSO):


    def _tf_data_gen(self):

        data_generator = iter(self.data_reader)

        while True:

            try:
                image_data: StereoImageData = next(data_generator)
            except StopIteration:
                return

            # prepare data
            yield (self.image_pre_proc(image_data.image_data_left.image),
                   self.image_pre_proc(image_data.image_data_right.image),
                   image_data.image_data_left.camera_calibration.camera_resolution,
                   image_data.image_data_left.camera_calibration.camera_intrinsic,
                   np.array([image_data.stereo_baseline] + [0.]*5),
                   image_data.image_data_left.image_id)

    def _get_tf_dataset_signature(self):

        image_shape = (*self.data_reader.image_size, 3)

        output_signature = [tf.TensorSpec(shape=image_shape, dtype=tf.float32, name='image_cam0'),
                            tf.TensorSpec(shape=image_shape, dtype=tf.float32, name='image_cam1'),
                            tf.TensorSpec(shape=(2,), dtype=tf.float32, name='cam_resolution'),
                            tf.TensorSpec(shape=(3, 3), dtype=tf.float32, name='cam_intrinsic'),
                            tf.TensorSpec(shape=(6,), dtype=tf.float32, name='cam_distance'),
                            tf.TensorSpec(shape=(), dtype=tf.int32, name='frame_id')]

        return output_signature




if __name__ == '__main__':

    _batch_size = 4
    _image_size = (768, 896)
    _path_data_dir = '/media/juri/T7/rawDatasets/baslerStereo/20250811_TUHH/20250811_TUHH_pallet'
    _data_reader = DataReaderForDVSO(path_data_dir=_path_data_dir, batch_size=_batch_size, image_size=_image_size)
    _tf_data_reader = _data_reader.to_tf_dataset()

    for _data in _tf_data_reader: # _data_reader._tf_data_gen()   _tf_data_reader:
        print(':)')
