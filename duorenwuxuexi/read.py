import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# 指定日志文件所在的目录
logdir = "separate_3"

# 创建一个 EventAccumulator 实例
ea = event_accumulator.EventAccumulator(logdir)
ea.Reload()  # 读取数据

# 获取所有的 tags
tags = ea.Tags()['scalars']
print("Tags available:", tags)

# 读取 'train_loss' 数据
readdd= ea.Scalars('val_acc1')

# tags = ["train_loss", "train_acc1", "train_acc2", "train_acc3", "val_loss", "val_acc1", "val_acc2", "val_acc3",
#         "learning_rate", "loss1", "loss2", "loss3", "w1", "w2", "w3", ]

# 将数据整理为一个字典，便于创建 DataFrame
data = {'epoch': [], 'val_acc1': []}
for entry in readdd:
    data['epoch'].append(entry.step)
    data['val_acc1'].append(entry.value)

# 将字典转换为 DataFrame
df = pd.DataFrame(data)

# 保存到 Excel 文件
output_file = "readdata.xlsx"
df.to_excel(output_file, index=False, engine='openpyxl')

print(f"Data saved to {output_file}")