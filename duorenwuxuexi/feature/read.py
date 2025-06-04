import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# 指定日志文件所在的目录
logdir = "ctq_mtl_test"

# 创建一个 EventAccumulator 实例
ea = event_accumulator.EventAccumulator(logdir)
ea.Reload()  # 读取数据

# 获取所有的 tags
tags = ea.Tags()['scalars']
print("Tags available:", tags)

# 读取 'train_loss' 数据
readdd= ea.Scalars('t_2_2')

# tags = ["a_v", "a_t", "v_2_1", "v_2_2", "v_2_3", "v_5_1", "v_5_2", "v_5_3", "t_2_1", "t_2_2", "t_2_3", "t_5_1", "t_5_2",
#         "t_5_3"]

# 将数据整理为一个字典，便于创建 DataFrame
data = {'epoch': [], 't_2_2': []}
for entry in readdd:
    data['epoch'].append(entry.step)
    data['t_2_2'].append(entry.value)

# 将字典转换为 DataFrame
df = pd.DataFrame(data)

# 保存到 Excel 文件
output_file = "readdata.xlsx"
df.to_excel(output_file, index=False, engine='openpyxl')

print(f"Data saved to {output_file}")