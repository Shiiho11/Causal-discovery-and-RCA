import pandas as pd
from datetime import datetime, timedelta

# 定义开始时间和结束时间
start_time = datetime(2021, 7, 1, 9, 57, 0)
end_time = datetime(2021, 8, 1, 0, 0, 0)

# 生成时间序列，步长为10秒
time_range = pd.date_range(start=start_time, end=end_time, freq='10S')

# 创建DataFrame
df_empty = pd.DataFrame({
    'time': time_range,
    'dbservice': float('nan'),
    'logservice': float('nan'),
    'mobservice': float('nan'),
    'redisservice': float('nan'),
    'webservice': float('nan')
})

print(df_empty)

name_list = ['dbservice', 'logservice', 'mobservice', 'redisservice', 'webservice']
file_name_list = ['dbservice.csv', 'logservice.csv', 'mobservice.csv', 'redisservice.csv', 'webservice.csv']
path = 'C:/Data/dataset/casual/GAIA/avg10s/'
for i in range(len(name_list)):
    name = name_list[i]
    file_name = file_name_list[i]
    print(name)
    df = pd.read_csv(path + file_name)
    data = df.values
    for row in data:
        t = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        v = row[1]
        row_number = (t - start_time) // timedelta(seconds=10)
        df_empty.at[row_number, name] = v
        # print(t)
        # print(row_number)
        # print(df_empty)
        # df_empty.to_csv(path+'test.csv')
        # exit()

print(df_empty)
df_empty.to_csv(path+'test.csv', index=False)
