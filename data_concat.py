import pandas as pd
import glob

data_paths = glob.glob('./resources/*.csv')
print(data_paths)
df = pd.DataFrame()
for data_path in data_paths:
    df_temp = pd.read_csv(data_path)
    df = pd.concat([df, df_temp])

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)  # 컬럼 번호 초기화
print(df.head())
print(df.tail())
df.info()
df.to_csv('./resources/test_data_final.csv', index=False)
