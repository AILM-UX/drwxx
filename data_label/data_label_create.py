import pandas as pd

# 读取xlsx文件
df = pd.read_excel('data_in.xlsx', header=None, names=['col1', 'col2'])

# 提取每行的文件名前缀
df['prefix'] = df['col1'].str.split('/').str[0]

# 创建一个空字典来存储每个前缀中形容词的出现次数
adjectives_counts = {}

# 遍历每行，统计每个前缀中形容词的出现次数
for index, row in df.iterrows():
    prefix = row['prefix']
    adjectives = row['col2'].split(', ')
    if prefix not in adjectives_counts:
        adjectives_counts[prefix] = {}
    for adj in adjectives:
        if adj in adjectives_counts[prefix]:
            adjectives_counts[prefix][adj] += 1
        else:
            adjectives_counts[prefix][adj] = 1

# 为每个前缀创建一个空列表来存储前五个形容词
top_adjectives = {prefix: [] for prefix in adjectives_counts}

# 对每个前缀的形容词字典按值（出现次数）进行排序，然后取前五个形容词
for prefix, adj_counts in adjectives_counts.items():
    sorted_adj_counts = sorted(adj_counts.items(), key=lambda x: x[1], reverse=True)
    for adj, count in sorted_adj_counts[:4]:
        top_adjectives[prefix].append(adj)

# 将结果转换为DataFrame
result = pd.DataFrame(list(top_adjectives.items()), columns=['prefix', 'col2'])

# 将结果保存到新的xlsx文件中
result.to_excel('data_out.xlsx', index=False)
result.to_excel('des.xlsx', index=False)

#材料/质地描述、外观/形状描述、表面特征描述、视觉特征描述、触觉特征描述、其他描述六个角度