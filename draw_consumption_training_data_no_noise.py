import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

CourseFolder=r""

home_number = 5
season_name = "winter"

time_stamp = pd.read_csv(CourseFolder+'occupancy_dataset/home' + str(home_number) + '/' + str(home_number).zfill(2) + '_' + season_name + '.csv')
plug = '01'
f = '2013-01-03.csv'

readings = pd.read_csv(CourseFolder+'occupancy_dataset/home' + str(home_number) + '/plugs/' + plug + '/' + f, header=None)
readings.insert(loc=0, column='time_stamp', value=time_stamp.columns.values.tolist()[1:])
readings = readings.rename({0: "electricity consumption"}, axis='columns')
readings['total electricity consumption'] = readings['electricity consumption'].cumsum()

hour_start = 1
minute_start = 20
second_start = 30
hour_end = 1
minute_end = 25
second_end = 30
start_time = hour_start*60*60+minute_start*60+second_start
end_time = hour_end*60*60+minute_end*60+second_end

plt.figure(1)
readings.iloc[start_time:end_time,:].plot(x='time_stamp', y='electricity consumption', figsize=(10,5))
# readings.plot(x='time_stamp', y='electricity consumption')
plt.xlabel('time')
plt.ylabel('electricity consumption')
plt.title('electricity consumption for home 5 (winter)')
plt.savefig('current consumption.png')
plt.show()

hour_start = 1
minute_start = 20
second_start = 30
hour_end = 2
minute_end = 50
second_end = 30
start_time = hour_start*60*60+minute_start*60+second_start
end_time = hour_end*60*60+minute_end*60+second_end

plt.figure(2)
readings.iloc[start_time:end_time,:].plot(x='time_stamp', y='total electricity consumption', figsize=(10,5))
# readings.plot(x='time_stamp', y='total electricity consumption')
plt.xlabel('time')
plt.ylabel('total electricity consumption')
plt.title('total electricity consumption graph for home 5 (winter)')
plt.savefig('cumulative consumption.png')
plt.show()
