from hoboreader import HoboReader
from windrose import WindroseAxes
from windrose import WindAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np

from pathlib import Path

loadPath = Path.cwd()
import seaborn as sns
from scipy import stats
import numpy as np
from tqdm import tqdm

def plotWR(ws, wd):
    ax = WindroseAxes.from_ax()
    ax.contourf(wd, ws, bins=np.arange(0, 6, 1), nsector=36, cmap=cm.hot)
    ax.contour(wd, ws, bins=np.arange(0, 6, 1),
               nsector=36, colors='black', lw=1)
    ax.set_legend()


def printPDF(ws):
    ax = WindAxes.from_ax()
    bins = np.arange(0, 6 + 1, 0.5)
    bins = bins[1:]
    ax, params = ax.pdf(ws, bins=bins)




S1 = HoboReader(loadPath / "DL1_Stone_Garden_Tjaden_Hall.csv")
S2 = HoboReader(loadPath / "DL2_PSB_2020.csv")
S3 = HoboReader(loadPath / "DL2_PSB_Parking_Lot.csv")
S4 = HoboReader(loadPath / "DL3_Olin_Library.csv")


df1 = S1.get_dataframe()
df2 = S2.get_dataframe()
df3 = S3.get_dataframe()
df4 = S4.get_dataframe()

# Variable to plot
variable1 = "Wind Speed"
variable2 = "Wind Direction"

# df1[variable].hist()
sns.distplot(df1[variable1], kde=False, fit=stats.gamma)
sns.distplot(df2[variable1], kde=False, fit=stats.gamma)
sns.distplot(df3[variable1], kde=False, fit=stats.gamma)
sns.distplot(df4[variable1], kde=False, fit=stats.gamma)

#sns.jointplot(x=variable1, y=variable2, data=df1);


list = [df1, df2, df3, df4]

# Variable to plot
variable1 = "Wind Speed"
variable2 = "Wind Direction"

ws = np.array(df1[variable1]).flatten()
wd = np.array(df1[variable2]).flatten()

for i in tqdm(list):
    ws = np.array(i[variable1]).flatten()
    wd = np.array(i[variable2]).flatten()
    plotWR(ws, wd)
    printPDF(ws)








