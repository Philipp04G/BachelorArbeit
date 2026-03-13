import numpy as np
from functools import partial
from pyproj import Transformer

def gps_coords_to_meter_old(latitude, longitude, altitude):
    # convert gps longitude and latitude to meters
    # TODO: improve formular (this one assumes earth is a sphere)
    # https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters
    latitude_meter = latitude * 111320
    longitude_meter = longitude * 40075000 * np.cos(latitude * np.pi /180) / 360
    gps_position_meter = np.array([latitude_meter, longitude_meter, altitude])
    return gps_position_meter

def meter_to_gps_coords_old(latitude_meter, longitude_meter, altitude):
    latitude = latitude_meter / 111320
    longitude = longitude_meter / (40075000 * np.cos(latitude * np.pi /180) / 360)
    gps_position = np.array([latitude, longitude, altitude])
    return gps_position


# some stuff for coordinate transformation
transformer_angle_to_meter = Transformer.from_crs(4326, 31468)
transformer_meter_to_angle = Transformer.from_crs(31468, 4326)

def gps_coords_to_meter(latitude, longitude, altitude):
    x, y, z = transformer_angle_to_meter.transform(latitude, longitude, altitude)
    return np.array([x, y, z])

def meter_to_gps_coords(latitude_meter, longitude_meter, altitude):
    lat, lon, att = transformer_meter_to_angle.transform(latitude_meter, longitude_meter, altitude)
    return np.array([lat, lon, att])
