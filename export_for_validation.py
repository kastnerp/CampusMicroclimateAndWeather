from helper_functions import *
#Export

y = 2019
m = 1
d = 1

yy = y
mm = 12
dd = 31

start_time = str(y)+'-'+str(m)+'-'+str(d)+' 07:00:00-00:00'
end_time = str(yy)+'-'+str(mm)+'-'+str(dd)+' 23:00:00-00:00'

df = S2.get_dataframe()['MRT']
df_2019 = df.loc[start_time:end_time].apply(lambda x: ftoc(x), axis=1).resample('H').mean()
df_2019.index = df_2019.index.map(lambda x: to_hour_of_year(x.hour, x.day, x.month))

df_2019.to_csv('ValidationTest/mrt.csv',index_label = ['hour_of_year'], header = [ 'MRT'])

y = 2019
m = 1
d = 1

yy = y
mm = 12
dd = 31

start_time = str(y)+'-'+str(m)+'-'+str(d)+' 07:00:00-00:00'
end_time = str(yy)+'-'+str(mm)+'-'+str(dd)+' 23:00:00-00:00'

df = S2.get_dataframe()['Solar Radiation']
df_2019 = df.loc[start_time:end_time].apply(lambda x: ftoc(x), axis=1).resample('H').mean()
df_2019.index = df_2019.index.map(lambda x: to_hour_of_year(x.hour, x.day, x.month))

df_2019.to_csv('ValidationTest/solar_radiation.csv',index_label = ['hour_of_year'], header = [ 'Solar Radiation'])