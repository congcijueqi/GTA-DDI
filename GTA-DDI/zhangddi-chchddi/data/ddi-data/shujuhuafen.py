import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据集
data_path = '/home/dwj/BIBM/SSI-DDI1-topk-zhangddi/data/ddi-data/all_data_zhangddi.csv'
df = pd.read_csv(data_path)

# 按照6:2:2的比例划分数据集
train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 保存划分后的数据集
train_data.to_csv('/home/dwj/BIBM/SSI-DDI1-topk-zhangddi/data/ddi-data/ZhangDDI_train.csv', index=False)
valid_data.to_csv('/home/dwj/BIBM/SSI-DDI1-topk-zhangddi/data/ddi-data/ZhangDDI_valid.csv', index=False)
test_data.to_csv('/home/dwj/BIBM/SSI-DDI1-topk-zhangddi/data/ddi-data/ZhangDDI_test.csv', index=False)

print("Data has been split and saved successfully.")
