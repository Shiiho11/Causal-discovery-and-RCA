import pandas as pd


def merge(path, file1, file2, output):
    # 步骤1: 读取第一个CSV文件
    df1 = pd.read_csv(path + file1)

    # 步骤2: 读取第二个CSV文件
    df2 = pd.read_csv(path + file2)

    # 步骤3: 合并两个DataFrame，这里使用concat函数进行垂直合并（假设两个文件的列结构相同）
    # 如果列名有差异或需要更精细的合并，请根据实际情况调整concat的参数
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # 步骤4: 确保'Time'列是日期时间类型，如果不是，需要转换
    if combined_df['timestamp'].dtype != 'datetime64[ns]':
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])

    # 步骤5: 按照时间列排序
    sorted_df = combined_df.sort_values(by='timestamp')

    # 步骤6: 可选：保存排序后的结果到新的CSV文件
    sorted_df.to_csv(path + output, index=False)


path = 'C:/Data/dataset/casual/GAIA/timedelta/'
merge(path, 'dbservice1.csv', 'dbservice2.csv', 'new_dbservice.csv')
merge(path, 'logservice1.csv', 'logservice2.csv', 'new_logservice.csv')
merge(path, 'mobservice1.csv', 'mobservice2.csv', 'new_mobservice.csv')
merge(path, 'redisservice1.csv', 'redisservice2.csv', 'new_redisservice.csv')
merge(path, 'webservice1.csv', 'webservice2.csv', 'new_webservice.csv')
