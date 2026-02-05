import numpy as np


def gps_coords_to_meter(latitude, longitude, altitude):
    # convert gps longitude and latitude to meters
    # TODO: improve formular (this one assumes earth is a sphere)
    # https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters
    latitude_meter = latitude * 111320
    longitude_meter = longitude * 40075000 * np.cos(latitude * np.pi /180) / 360
    gps_position_meter = np.array([latitude_meter, longitude_meter, altitude])
    return gps_position_meter

def meter_to_gps_coords(latitude_meter, longitude_meter, altitude):
    latitude = latitude_meter / 111320
    longitude = longitude_meter / (40075000 * np.cos(latitude * np.pi /180) / 360)
    gps_position = np.array([latitude, longitude, altitude])
    return gps_position