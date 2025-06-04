import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('data_label.xlsx', header=0)

# 填充空白单元格为空字符串
df.fillna('', inplace=True)

# 获取所有形容词列，排除 'prefix' 列
adjectives = df.iloc[:, 1:].values.flatten()

# 统计每个形容词的出现次数
adjective_counts = pd.Series(adjectives).value_counts()

# 过滤掉出现次数少于5次的形容词
common_adjectives = adjective_counts[adjective_counts >= 8].index

# 定义一个函数，检查形容词是否在常见形容词列表中，不在则替换为空字符串
def filter_adjectives(x):
    return x if x in common_adjectives else ''

# 应用过滤函数到 DataFrame 的所有形容词列
filtered_df = df.iloc[:, 1:].applymap(filter_adjectives)

# 将 'prefix' 列加回 DataFrame
filtered_df.insert(0, 'prefix', df['prefix'])

# 保存处理后的 DataFrame 到新文件
filtered_df.to_excel('filtered_data_label.xlsx', index=False)

print("处理完毕，保存结果到 'filtered_data_label.xlsx'")



# 读取xlsx文件
df = pd.read_excel('filtered_data_label.xlsx', header=None)

# 去掉所有单元格中的单引号
df = df.applymap(lambda x: x.strip("'") if isinstance(x, str) else x)

# 保存为新文件
df.to_excel('cleaned_data_label.xlsx', index=False, header=False)
import pandas as pd

# 读取包含形容词和标签的两个表格
adjectives_df = pd.read_excel('cleaned_data_label.xlsx')
labels_df = pd.read_excel('label1.xlsx')

# 创建一个字典，将形容词映射到标签
adjective_to_label = dict(zip(labels_df['adjective'], labels_df['label']))

# 遍历第一张表格的每一列
for col in adjectives_df.columns[1:]:
    # 将每个形容词替换为对应的标签，并去除小数点
    adjectives_df[col] = adjectives_df[col].fillna("").apply(lambda x: adjective_to_label.get(str(x).rstrip('.'), x))

# 保存结果到新的表格中
adjectives_df.to_excel('replaced_adjectives.xlsx', index=False)
import pandas as pd

# 读取原始表格
df = pd.read_excel('replaced_adjectives.xlsx')
# 创建新的DataFrame来存储结果

# 创建一个空列表来存储每一行的结果
rows = []

# 遍历原始表格的每一行
for index, row in df.iterrows():
    # 创建一个列表来存储当前行的标签
    # labels = ['0'] * 6
    labels = ['0'] * 3

    # 遍历当前行的标签
    for label in row.values[1:]:
        # 检查标签是否是浮点数，如果是，则转换为字符串类型

        label = str(label)

        # 检查标签是否具有正确的格式
        if '_' in label:
            # 找到第一个出现的1_到6_的标签
            # if label.startswith('1_') or label.startswith('2_') or label.startswith('3_') or label.startswith('4_') or label.startswith('5_') or label.startswith('6_') :
            if label.startswith('1_') or label.startswith('2_') or label.startswith('3_')  :

                # 提取标签中的数字部分作为索引
                index_str = label.split('_')[0]
                if index_str.isdigit():
                    index = int(index_str) - 1
                    index_label=label.split('_')[1]
                    # 更新标签列表中对应索引位置的值
                    labels[index] = index_label

    # 将结果添加到行列表中
    rows.append({'prefix': row['prefix'],
                 'label1': labels[0],
                 'label2': labels[1],
                 'label3': labels[2],
                 })
    # rows.append({'prefix': row['prefix'],
    #              'label1': labels[0],
    #              'label2': labels[1],
    #              'label3': labels[2],
    #              'label4': labels[3],
    #              'label5': labels[4],
    #              'label6': labels[5]
    #              })


# 使用行列表创建新的 DataFrame
result_df = pd.DataFrame(rows)

# 保存结果到新的表格中
result_df.to_excel('label_output.xlsx', index=False)

import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('label_output.xlsx')
# 定义一个空字典来存储每列的标签到编号的映射
label_to_id_map = {}

# 遍历每一列
for col in df.columns[1:]:
    # 获取该列的唯一标签，并按照大小顺序重新编号
    unique_labels = sorted(df[col].unique())
    label_to_id = {label: i for i, label in enumerate(unique_labels)}

    # 将该列的标签到编号的映射存储到字典中
    label_to_id_map[col] = label_to_id

    # 使用重新编号后的标签替换原始标签
    df[col] = df[col].map(label_to_id)

# 保存为新的 Excel 文件
df.to_excel('label_final_output.xlsx', index=False)