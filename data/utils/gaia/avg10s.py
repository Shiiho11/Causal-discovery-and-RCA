import pandas as pd


# def run(path, file_name):
#     # 读取CSV文件
#     df = pd.read_csv(path + file_name)
#
#     # 确保时间列是datetime类型
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#
#     # 设置一个新的时间列，向下取整到最近的10秒间隔的开始时间
#     df['Interval Start'] = df['timestamp'].dt.floor('10S')
#
#     # 按照新的时间间隔列进行分组，并计算数值列的平均值
#     grouped = df.groupby('Interval Start')['timedelta'].mean().reset_index()
#
#     # 输出到新的CSV文件
#     grouped_full.to_csv(path + file_name + '.new', index=False)

def run(path, file_name):
    # 读取CSV文件
    df = pd.read_csv(path + file_name)

    # 确保时间列是datetime类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 设置一个新的时间列，向下取整到最近的10秒间隔的开始时间
    df['timestamp'] = df['timestamp'].dt.floor('10S')

    # 按照新的时间间隔列进行分组，并计算数值列的平均值
    grouped = df.groupby('timestamp')['timedelta'].mean().reset_index()

    # 步骤2: 创建新的df，里面是缺失时间的数据（空值填充）
    time_range = pd.date_range(start=df['timestamp'].min(),
                               end=df['timestamp'].max(), freq='10S')
    missing_times_df = pd.DataFrame({'timestamp': time_range, 'timedelta': float('nan')})
    missing_times_df = missing_times_df[~missing_times_df['timestamp'].isin(grouped['timestamp'])]

    # 步骤3: 将原始grouped数据与新创建的缺失时间数据concat
    combined_df = pd.concat([grouped, missing_times_df], ignore_index=True)

    # 步骤4: 按照时间排序新的df
    combined_df.sort_values(by='timestamp', inplace=True)

    # 输出到新的CSV文件
    combined_df.to_csv(path + file_name + '.new', index=False)


path = 'C:/Data/dataset/casual/GAIA/merge/'
file_name_list = ['dbservice.csv', 'logservice.csv', 'mobservice.csv', 'redisservice.csv', 'webservice.csv']
for file_name in file_name_list:
    run(path, file_name)
