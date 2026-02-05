#import rosbag
#from sensor_msgs.point_cloud2 import read_points
from samson_data_reader.raw_data_reader.abc_raw_data_reader import RawDataReader
from samson_data_reader.datacalsses import *
from samson_data_reader.utils.gps_conversion import gps_coords_to_meter

import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path


def _find_bag_path(base: str | Path) -> Path:
    base = Path(base)
    if base.is_file() and base.suffix == '.bag':
        return base  # direkt auf .bag gezeigt (ROS1)

    # ROS1: irgendeine *.bag im Ordner
    ros1 = sorted(base.glob('*.bag'), key=lambda p: p.stat().st_mtime, reverse=True)
    if ros1:
        return ros1[0]

    # ROS2: Ordner mit metadata.yaml
    ros2 = [d for d in base.iterdir() if d.is_dir() and (d / 'metadata.yaml').exists()]
    if ros2:
        # nimm den jüngsten
        ros2.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return ros2[0]

    raise FileNotFoundError(f"Kein ROS-Bag gefunden in: {base}")


class RosBagReader(RawDataReader):

    #gps_topic = '/ublox/fix'
    gps_topic = '/gpsfix'
    lidar_topic = '/ouster/points'

    def __init__(self, path_rosbag, read_gps=True, read_lidar=True, **kwargs):

        print('Opening bag with rosbags (no ROS needed)…')
        #self.ros_bag = rosbag.Bag(path_rosbag)
        self.bag_path = _find_bag_path(path_rosbag)
        self.reader = AnyReader([self.bag_path])
        self.reader.__enter__()
        print('Finished reading ros bag file.')



        available = [c.topic for c in self.reader.connections]
        candidates = ['/ublox/fix', '/gpsfix']
        self.gps_topic = next((t for t in candidates if t in available), None)
        self.ros_topics = []
        if self.gps_topic:
            self.ros_topics.append(self.gps_topic)
        else:
            print(f"[RosBagReader] No GPS topic found. Available: {available}")
        # self.ros_topics = []
        # if read_gps:
        #     self.ros_topics.append(self.gps_topic)
        # if read_lidar:
        #     self.ros_topics.append(self.lidar_topic)

        # Vorbereiten: Connections nach Topics
        self._conns = [c for c in self.reader.connections if c.topic in self.ros_topics]
        self._msg_iters = None
        self._next_ros_message = None # wird in __iter__ gefüllt


    def next_time(self):
        if self._next_ros_message is None:
            self._next_ros_message = self._read_ros_message()  # darf None liefern, wenn leer
            # Wir kennen die nächste Zeit (noch) nicht -> „unendlich“,
            # damit andere Reader den Vortritt bekommen.
        if self._next_ros_message is None:
            return float('inf')
        return self._next_ros_message.time


    def __iter__(self):
        # self.ros_bag_gen = self.ros_bag.read_messages(topics=self.ros_topics)

        # # read the next message, to the next time is known
        # self._next_ros_message = self._read_ros_message()

        # Baue pro Connection einen Iterator; kein Vorab-Read!
        self._msg_iters = [self.reader.messages(connections=[c]) for c in self._conns]
        # NICHT vorlesen; _next_ros_message bleibt None bis zum ersten __next__()
        self._next_ros_message = None
        return self


    def __next__(self):
        # Falls nichts gepuffert: jetzt eine Nachricht ziehen
        if self._next_ros_message is None:
            self._next_ros_message = self._read_ros_message()

        # Wenn weiterhin nichts: Ende der Iteration
        if self._next_ros_message is None:
            raise StopIteration

        current_ros_message = self._next_ros_message
        # gleich die nächste vorbereiten (kann None werden)
        self._next_ros_message = self._read_ros_message()
        return current_ros_message



    def _read_ros_message(self):
        # try:
        #     topic, ros_message, ros_timestamp = next(self.ros_bag_gen)
        # except StopIteration:
        #     raise StopIteration

        # # TODO: convert ros message to dataclass
        # if topic == self.gps_topic:
        #     return self.convert_gps_message(ros_message, ros_timestamp)
        # elif topic == self.lidar_topic:
        #     return self.convert_lidar_message(ros_message, ros_timestamp)
        # else:
        #     raise AttributeError('Unexpected ros topic')
        # Nächste Nachricht je Topic holen (ohne zu verlieren)
            # sammle „nächste“ Kandidaten aus allen Iteratoren (per Peek/Copy oder Try-Next mit Cache)
        candidates = []
        for it, conn in zip(self._msg_iters, self._conns):
            try:
                conn, ts, raw = next(it)
                candidates.append((ts, conn, raw))
            except StopIteration:
                continue
        if not candidates:
            return None
        
                # wenn du peeken willst, brauchst du Cache. Einfacher: konsumieren und optional zurücklegen.
                # Hier der einfache Weg: wir nehmen die erste gefundene, oder du sortierst nach ts.
        ts_min, conn_min, raw_min = min(candidates, key=lambda x: x[0])
        msg = self.reader.deserialize(raw_min, conn_min.msgtype)
        # dispatch
        if conn_min.topic in ('/ublox/fix', '/gpsfix'):
            return self.convert_gps_message(msg, ts_min, topic=conn_min.topic)

        elif conn_min.topic == self.lidar_topic:
            return self.convert_lidar_message(msg, ts_min)

        else:
            # andere Topics (z.B. /imu/data) ignorieren, nächste Nachricht holen
            return self._read_ros_message()



    def convert_gps_message(self, ros_message, ros_timestamp_ns, topic='/ublox/fix'):

        # # convert the ros timestamp to a unix timestamp
        # timestamp = ros_timestamp.secs + ros_timestamp.nsecs * 1e-9

        # gps_position = gps_coords_to_meter(ros_message.latitude, ros_message.longitude, ros_message.altitude)

        # # TODO: convert the gps covariance to meter?
        # gps_covar = np.reshape(np.array(ros_message.position_covariance), (3, 3))

        # return GpsData(
        #     time=timestamp,
        #     position=gps_position,
        #     position_variance=gps_covar
        # )
        # rosbags: ts ist Nanosekunden
            t = float(ros_timestamp_ns) * 1e-9
            if topic == '/ublox/fix':   # sensor_msgs/NavSatFix
                lat, lon, alt = float(ros_message.latitude), float(ros_message.longitude), float(ros_message.altitude)
                cov = getattr(ros_message, 'position_covariance', [0.0]*9)
            else:                       # gps_common/GPSFix
                lat, lon, alt = float(ros_message.latitude), float(ros_message.longitude), float(ros_message.altitude)
                cov = getattr(ros_message, 'position_covariance', [0.0]*9)
            pos = gps_coords_to_meter(lat, lon, alt)
            cov3 = np.reshape(np.array(cov, dtype=np.float32), (3,3))
            return GpsData(time=t, position=pos, position_variance=cov3)

    def convert_lidar_message(self, ros_message, ros_timestamp_ns):

        # # convert the ros timestamp to a unix timestamp
        # timestamp = ros_timestamp.secs + ros_timestamp.nsecs * 1e-9

        # # decide whether to use previous or next lidar scan
        # points, points_intensities, points_timestamps = self._read_pointcloud2(ros_message)
        # points_timestamps += timestamp

        # # remove invalid points
        # points_valid = np.any(points != 0, axis=1)
        # points = points[points_valid]
        # points_timestamps = points_timestamps[points_valid]

        # return LidarPointsData(
        #     time=timestamp,
        #     points=points,
        #     points_time=points_timestamps
        # )
        timestamp = float(ros_timestamp_ns) * 1e-9

        points, intensities, time_offsets = self._read_pointcloud2(ros_message)
        points_timestamps = time_offsets + timestamp

        # ungültige raus
        points_valid = np.any(points != 0, axis=1)
        points = points[points_valid]
        points_timestamps = points_timestamps[points_valid]

        return LidarPointsData(
            time=timestamp,
            points=points,
            points_time=points_timestamps
        ) 

    @staticmethod
    def _read_pointcloud2(msg):

        # x_list, y_list, z_list, intensity_list, time_offset_list = [], [], [], [], []
        # for data_point in read_points(ros_message):

        #     x, y, z, intensity, t, reflectivity, ring, ambient, point_range = data_point
        #     # TODO: convert the time offset.
        #     #  ATTENTION: This conversion is guessed
        #     time_offset = -t * 1e-9

        #     # add data to lists
        #     x_list.append(x)
        #     y_list.append(y)
        #     z_list.append(z)
        #     intensity_list.append(intensity)
        #     time_offset_list.append(time_offset)

        # # convert the lists to numpy arrays
        # points = np.stack([x_list, y_list, z_list], axis=-1)
        # intensities = np.array(intensity_list)
        # timestamps = np.array(time_offset_list)

        # return points, intensities, timestamps
        import struct
        n = msg.width * msg.height
        if n == 0:
            return (np.zeros((0,3), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.float32))

        # Field-Offsets ermitteln
        offs = {f.name: f.offset for f in msg.fields}
        has_intensity = 'intensity' in offs
        has_time = 'time' in offs or 't' in offs  # manche Ouster-Bags nutzen 't'
        tkey = 'time' if 'time' in offs else ('t' if 't' in offs else None)

        step = msg.point_step
        data = memoryview(msg.data)
        pts = np.empty((n, 3), dtype=np.float32)
        inten = np.zeros((n,), dtype=np.float32)
        toff = np.zeros((n,), dtype=np.float32)

        # schnelle vektorlose, aber robuste Schleife
        for i in range(n):
            base = i * step
            pts[i, 0] = struct.unpack_from('f', data, base + offs['x'])[0]
            pts[i, 1] = struct.unpack_from('f', data, base + offs['y'])[0]
            pts[i, 2] = struct.unpack_from('f', data, base + offs['z'])[0]
            if has_intensity:
                inten[i] = struct.unpack_from('f', data, base + offs['intensity'])[0]
            if tkey:
                toff[i] = struct.unpack_from('f', data, base + offs[tkey])[0]

        # Dein Original hat time_offset = -t * 1e-9 geraten; wir lassen toff in Sekunden:
        # Viele Ouster-Themen kodieren t in ns → skaliere, falls sehr groß:
        if np.nanmax(np.abs(toff)) > 1e6:
            toff = toff * 1e-9  # ns → s

        return pts, inten, toff
    
    def __del__(self):
        try:
            self.reader.__exit__(None, None, None)
        except Exception:
            pass





if __name__ == '__main__':

    _bag_path = '/media/juri/T7/rawDatasets/SAMSON_IFAM_Basler/20240821_ApfelGroese/2024-08-21_10-17-12_A27_Groeßenmessung_vorher.bag'
    _data_reader = RosBagReader(path_rosbag=_bag_path)

    for _data in _data_reader:
        print(':)')