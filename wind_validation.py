import numpy as np
import math
import pandas as pd
from pandas import read_csv
import csv
import unittest
import numpy as np
from pprint import pprint
from setuptools import setup, find_packages
from codecs import open
from os import path
from epw import epw
from helper_functions import *
import pathlib

current_dir = pathlib.Path().absolute()


def radial_angle_from_north(vector_2):
    TrueNorth = [0, 1, 0]

    unit_vector_1 = TrueNorth
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle_rad = np.arccos(dot_product)

    angle_deg = math.degrees(angle_rad)

    return (angle_deg)


def offset(angle_1, angle_2):
    return (angle_2 - angle_1)


# vector 2 is simulated vector for the same station
StatVect = [0.779114, -2.236837, 0.337169]
SimulatedAngle = radial_angle_from_north(StatVect)
print(SimulatedAngle)

# second angle is the angle from the station recorded data
print(offset(SimulatedAngle, 115))


def load_epw_df(filepath):
    template = epw()
    # epw file path of the station we want
    template.read(filepath)
    df = template.dataframe
    return df


epw_file = load_epw_df(current_dir / "PSBLot_2020.epw")

wind_df = epw_file["Wind Direction"]


def all_hours_constant(df, max_y_interval):
    res = df.between(-max_y_interval, max_y_interval)

    all_between = res.all()

    return all_between


def find_hours_w_constant_wind_dirs(max_y_interval, requested_x_interval, wind_dir_df):
    # requested_x_interval - hours wind is constant
    # max_y_interval - max difference allowed while considered to be constant

    # write this as forward counting method

    hours_constant = []

    wind_dir_diff = wind_df.diff()

    for hour, diff in enumerate(wind_dir_diff):
        period = wind_dir_diff[hour:hour + requested_x_interval]
        if (all_hours_constant(period, max_y_interval) and not (wind_dir_df[hour:hour + requested_x_interval] == 179).all()):
            hours_constant.append(hour)

    return hours_constant


h = find_hours_w_constant_wind_dirs(10, 8, wind_df)
# h will give you all hours of the year for which the following 8 hours are within +-10° of wind direction in either direction
