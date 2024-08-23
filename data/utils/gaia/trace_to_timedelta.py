import time

import numpy as np
import pandas as pd
from datetime import datetime


def auto_add_decimal(date_str):
    if '.' not in date_str:
        date_str += '.0'
    return date_str


def trace_to_timedelta(path, file_name):
    df = pd.read_csv(path + file_name, usecols=['timestamp', 'start_time', 'end_time'])
    data = df.values
    new_data = np.full((len(data), 2), 0, dtype=object)
    for i in range(len(data)):
        new_data[i][0] = datetime.strptime(data[i][0], "%Y-%m-%d %H:%M:%S")
        td = (datetime.strptime(auto_add_decimal(data[i][2]), "%Y-%m-%d %H:%M:%S.%f") -
              datetime.strptime(auto_add_decimal(data[i][1]),"%Y-%m-%d %H:%M:%S.%f"))
        new_data[i][1] = td.total_seconds()
    new_df = pd.DataFrame(new_data, columns=['timestamp', 'timedelta'])
    new_df.to_csv(path + file_name + '.new', index=False)


path = 'C:/Data/dataset/casual/GAIA/origin/'
file_name_list = ['trace_table_dbservice1_2021-07.csv',
                  'trace_table_dbservice2_2021-07.csv',
                  'trace_table_logservice1_2021-07.csv',
                  'trace_table_logservice2_2021-07.csv',
                  'trace_table_mobservice1_2021-07.csv',
                  'trace_table_mobservice2_2021-07.csv',
                  'trace_table_redisservice1_2021-07.csv',
                  'trace_table_redisservice2_2021-07.csv',
                  'trace_table_webservice1_2021-07.csv',
                  'trace_table_webservice2_2021-07.csv']
t = time.time()
for file_name in file_name_list:
    print(file_name)
    trace_to_timedelta(path, file_name)
print('time:', time.time()-t)
