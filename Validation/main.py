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



def radial_angle(vector_1, vector_2) :

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle_rad = np.arccos(dot_product)

    angle_deg = math.degrees(angle_rad)

    return(angle_deg)

#vector 1 = true NORTH
TrueNorth = [0, 1, 0]
#vector 2 is simulated vector for the same station
StatVect = [0.779114, -2.236837, 0.337169]
SimulatedAngle = radial_angle(TrueNorth, StatVect)
print (SimulatedAngle)

def offset(angle_1, angle_2) :
    return(angle_2 - angle_1)

# second angle is the angle from the station recorded data
print(offset(SimulatedAngle, 115))

def to_hour_of_year(hour, day, month):
    # count from 1
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    m = sum(days_in_month[:month - 1]) * 24
    d = (day - 1) * 24
    h = hour

    return m + d + h

template = epw()
# epw file path of the station we want
template.read(r"C:\Users\remym\Documents\GitHub\HoboPlot\PSBLot_2020.epw")
df = template.dataframe
df






