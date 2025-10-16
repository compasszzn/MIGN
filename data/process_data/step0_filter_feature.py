import os
import pandas as pd
from glob import glob
from datetime import datetime, timedelta

# 数据目录
base_dir = "/mnt/hda/zzn/realtime/Dataset"
output_dir = "/mnt/hda/zzn/realtime/nips"
from tqdm import tqdm
os.makedirs(output_dir, exist_ok=True)
def fahrenheit_to_kelvin(fahrenheit):
    return (5/9) * (fahrenheit - 32) + 273.15
# 变量及缺失值对应关系
variables = {
    # "VISIB": 999.9,
    # "MXSPD": 999.9,
    # "GUST": 999.9,
    # "STP": 9999.9,
    "DEWP": 9999.9,
    # "SLP": 9999.9, 
    # "PRCP": 99.99, "SNDP": 999.9, "WDSP": 999.9,
    # "MAX": 9999.9, "MIN": 9999.9, 
}

# 遍历所有年份
for year in range(2023, 2025):
    year_path = os.path.join(base_dir, str(year))
    if not os.path.exists(year_path):
        continue  # 跳过不存在的年份目录

    # 获取该年份所有气象站的 CSV 文件
    csv_files = glob(os.path.join(year_path, "*.csv"))
    # csv_files = csv_files[:5]

    # **从 1 月 1 日遍历到 12 月 31 日**
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    while start_date <= end_date:
        date_str = start_date.strftime("%Y-%m-%d")  # 生成 'YYYY-MM-DD' 格式
        daily_data = {var: [] for var in variables}  # 存储当天各个变量的数据

        for file in tqdm(csv_files):
            df = pd.read_csv(file, dtype={'STATION': str})

            # 过滤出当天的数据
            df = df[df['DATE'] == date_str]

            # 遍历所有变量，将有效数据归类
            for var, missing_value in variables.items():
                if var in df.columns:

                    df_var = df[df[var] != missing_value][['LONGITUDE', 'LATITUDE', 'STATION', var]]

                    # **优化：使用 values.tolist() 提升性能**
                    daily_data[var].extend(df_var.values.tolist())


        # **将当天数据写入文件，并释放内存**
        for var, records in daily_data.items():
            if not records:  # 如果该变量当天没有数据，跳过
                continue

            var_dir = os.path.join(output_dir, var)
            os.makedirs(var_dir, exist_ok=True)
            file_path = os.path.join(var_dir, f"{date_str}.csv")

            df_out = pd.DataFrame(records, columns=["longitude", "latitude", "station_id", "observation_value"])
            df_out = df_out.dropna(how='any')
            if var in ["MAX","MIN","TEMP","DEWP"]:
                df_out['observation_value'] = df_out['observation_value'].apply(fahrenheit_to_kelvin)
            df_out = df_out.drop_duplicates(subset=['station_id'], keep='first')
            # 追加模式，防止丢失数据
            df_out.to_csv(file_path, index=False)

        print(f"✅ {date_str} 处理完成")
        
        # **日期递增**
        start_date += timedelta(days=1)
