import pandas as pd
import re

# 读取xlsx文件
df = pd.read_excel('data_in.xlsx', header=None, names=['col1', 'col2'])

# 提取所有出现过的形容词
all_adjectives = []
for col2 in df['col2']:
    adjectives = re.findall(r'\b\w+\b', col2)  # 使用正则表达式提取所有单词
    all_adjectives.extend(adjectives)

# 去重并整理成一个形容词列表
all_adjectives = list(set(all_adjectives))

print(all_adjectives)
