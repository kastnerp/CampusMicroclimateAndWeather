import numpy as np
import math

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
from tqdm import tqdm
import pandas as pd
from hoboreader import HoboReader
from enum import Enum
import pathlib as pl
epw_path = pl.WindowsPath(r"C:\Users\pkastner\Documents\GitHub\HoboPlot")

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

def to_hour_of_year_dt(dt):
    # count from 1

    if calendar.isleap(dt.year):
        days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        print("Leap Year!")
    else:
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    m = sum(days_in_month[:int(dt.month) - 1]) * 24
    d = (int(dt.day) - 1) * 24
    h = int(dt.hour)

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


# h = find_hours_w_constant_wind_dirs(10, 8, wind_df)
# h will give you all hours of the year for which the following 8 hours are within +-10° of wind direction in either direction



def find_hours_w_constant_wind_dirs(wind_dir_interval, n_hours_constant,
                                    U_above, wind_dirs, wind_vels):
    # requested_x_interval - hours wind is constant
    # max_y_interval - max difference allowed while considered to be constant

    def all_hours_constant(s, max_y_interval):

        res = np.all(np.logical_and(s >= -max_y_interval, s <= max_y_interval))
        # print(res)
        return res.all()

    # all_hours_constant(np.array([4,4,3.5,4,5]), 15)

    def is_U_above(s, U_above):
        res = np.all(s > U_above)
        # print(res)
        return res

    # is_U_above(3,np.array([4,4,3.5,4,5]))

    def is_fill_data(s):
        res = np.all(s == 999)
        # print(res)
        return res

    hours_constant = []
    wind_dir_diff = np.diff(wind_dirs)
    total = len(wind_dir_diff)

    i = 0
    while  i < total:

        period_dirs = wind_dir_diff[i:i + n_hours_constant]
        period_vels = wind_vels[i:i + n_hours_constant]

        is_const = all_hours_constant(period_dirs, wind_dir_interval)
        is_above_Y = is_U_above(period_vels, U_above)
        is_filled = is_fill_data(period_dirs)

        if is_const and is_above_Y and not is_filled:
            hours_constant.append(i)
            i += n_hours_constant
        else:
            i  += 1
        #print(i)

    return hours_constant


# h = find_hours_w_constant_wind_dirs(10, 8, wind_df)
# h will give you all hours of the year for which the following 8 hours are within +-10° of wind direction in either direction
def get_1y_data(df, year):
    return df[str(year) + '-01-01':str(year) + '-12-31']



def get_meas_U(epw_name):
    ep = epw()
    ep.read(epw_path / epw_name)
    epw_df = ep.dataframe

    wind_speed = epw_df['Wind Speed']
    wind_dir = epw_df['Wind Direction']

    return wind_speed, wind_dir


def scale_abl(URefEPW, zref, z0, probing_height):
    zGround = 0
    Kappa = 0.41
    U_star = Kappa * URefEPW / (math.log((zref + z0) / z0))
    return U_star / Kappa * math.log((probing_height - zGround + z0) / z0)






class SeasonE(Enum):
    Winter = 1
    Spring = 2
    Summer = 3
    Fall = 4


class Season:
    def __init__(self, season_enum):
        self.DayBegin = 1
        if season_enum == SeasonE.Spring:

            self.DayEnd = 31
            self.MonthBegin = 3
            self.MonthEnd = 5
            self.YearBegin = 2021
            self.YearEnd = 2021

        elif season_enum == SeasonE.Summer:

            self.DayEnd = 31
            self.MonthBegin = 6
            self.MonthEnd = 8
            self.YearBegin = 2021
            self.YearEnd = 2021


        elif season_enum == SeasonE.Fall:
            self.DayEnd = 30
            self.MonthBegin = 9
            self.MonthEnd = 11
            self.YearBegin = 2021
            self.YearEnd = 2021

        else:  # Winter
            self.DayEnd = 28
            self.MonthBegin = 12
            self.MonthEnd = 2
            self.YearBegin = 2021
            self.YearEnd = 2022


def get_season_hour_bm(season_enum):

    s = Season(season_enum)

    season_hours = pd.date_range(str(s.YearBegin)+'-' + str(s.MonthBegin) + '-' + str(s.DayBegin) + 'T00:00:00.000Z',
                                 str(s.YearEnd)+'-' + str(s.MonthEnd) + '-' + str(s.DayEnd) + 'T23:00:00.000Z', freq='H')


    hours = [to_hour_of_year_dt(i) for i in season_hours]


    annual_hours = np.arange(0,8760,1)

    hours_bm = np.array([True if i  in hours else False for i in annual_hours ])

    return hours_bm




#len(get_season_hour_bm(SeasonE.Spring))


#S5 = HoboReader(r"C:\Users\pkastner\Documents\GitHub\HoboPlot\DL5_Game_Farm_Road.csv")
#df5 = S5.get_dataframe()
#df5 = df5.shift(periods=(-2), fill_value=0)
#df5_1y = df5
#wind_dirs = df5_1y.loc[:, ['Wind Direction']].values.flatten()
#wind_vels = df5_1y.loc[:, ['Wind Speed']].values.flatten()
#
## wind_dir_interval, n_hours_constant, U_above, wind_dirs, wind_vels
#h = find_hours_w_constant_wind_dirs(15, 4, 3.5, wind_dirs, wind_vels)
#
#df  = S5.get_dataframe()
#df_ds = df[['Wind Speed', 'Wind Direction', 'Date Time']].iloc[h]
#df_ds


