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
import regex as re
from epw import epw
import calendar
import pathlib
import tqdm

def windows_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def plot_graph(plt, start_time, variable):
    plt.savefig(windows_filename(start_time) + '_' + variable + '.pdf')


def ftoc(f):
    return (f - 32) * 5.0 / 9.0


def mphtoms(mph):
    return mph * 0.44704


def to_hour_of_year(hour, day, month, year):
    # count from 1

    if calendar.isleap(year):
        days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        print("Leap Year!")
    else:
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    m = sum(days_in_month[:int(month) - 1]) * 24
    d = (int(day) - 1) * 24
    h = int(hour)

    return m + d + h


def get_eval_hours(hour_start, hour_end, day_start, day_end, month_start,
                   month_end):
    # count from 1
    if (month_end > 12):
        month_end = 12
    if (day_end > 31):
        day_end = 31
    if (hour_end > 24):
        hour_end = 24

    cnt = 0
    hoursToEvaluate = []

    for m in range(12):  # 0-11
        for d in range(31):  # 0-30
            for h in range(24):  # 0-23

                # Check if already gone through month

                if (m == 2 and d > 27):
                    continue
                elif ((m == 4 or m == 6 or m == 9 or m == 10) and d > 29):
                    continue

                # Fill list
                cnt += 1

                if (m >= month_start and m < month_end and d >= day_start
                        and d < day_end and h >= hour_start and h < hour_end):
                    hoursToEvaluate.append(cnt)

    return hoursToEvaluate


# len(get_eval_hours(0, 24, 0, 31, 0, 12))


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
# StatVect = [0.779114, -2.236837, 0.337169]
# SimulatedAngle = radial_angle_from_north(StatVect)
# print(SimulatedAngle)

# second angle is the angle from the station recorded data
# print(offset(SimulatedAngle, 115))


def load_epw_df(filepath):
    template = epw()
    # epw file path of the station we want
    template.read(filepath)
    df = template.dataframe
    return df


# epw_file = load_epw_df(current_dir / "PSBLot_2020.epw")

# wind_df = epw_file["Wind Direction"]


def all_hours_constant(s, max_y_interval):
    res = s.between(-max_y_interval, max_y_interval)

    all_between = res.all()

    return all_between



def find_hours_w_constant_wind_dirs(max_y_interval, requested_x_interval, wind_dir_series):

    # requested_x_interval - hours wind is constant
    # max_y_interval - max difference allowed while considered to be constant

    # write this as forward counting method

    hours_constant = []

    wind_dir_series_diff = wind_dir_series.diff()

    for hour, diff in tqdm(enumerate(wind_dir_series_diff), total=wind_dir_series.shape[0]):
        period = wind_dir_series_diff[hour:hour + requested_x_interval]
        if (all_hours_constant(period, max_y_interval) and not (
                wind_dir_series[hour:hour + requested_x_interval] == 999).all()):
            hours_constant.append(hour)

    return hours_constant

# h = find_hours_w_constant_wind_dirs(10, 8, wind_df)
# h will give you all hours of the year for which the following 8 hours are within +-10° of wind direction in either direction
