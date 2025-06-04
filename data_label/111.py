import pickle
import os
import torch
from dataloador import vit_tac_Dataset
# 加载模型
    #,weights_only=False
with open(r'E:\BaiduNetdiskDownload\duorenwuxuexi(1)\data_label\t_v_dataset_train.pt', 'rb') as f:
    model = pickle.load(f)

def replace_paths(obj, old_path, new_path):
    """
    递归遍历对象，替换所有字符串中的旧路径为新路径
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = replace_paths(v, old_path, new_path)
        return obj
    elif isinstance(obj, list):
        return [replace_paths(item, old_path, new_path) for item in obj]
    elif isinstance(obj, str):
        # 替换路径（兼容不同斜杠格式）
        obj = obj.replace(old_path.replace('\\', '/'), new_path.replace('\\', '/'))  # 统一为 /
        obj = obj.replace(old_path.replace('/', '\\'), new_path.replace('/', '\\'))  # 统一为 \
        return obj
    else:
        return obj

old_path = "G:/dataset/tvl_dataset/hct/data"
new_path = "E:/duorenwuxuexi(1)/dataset/tvl_dataset/hct/data"

# 执行替换
modified_data = replace_paths(model, old_path, new_path)

# os.rename(
#     r'E:\BaiduNetdiskDownload\duorenwuxuexi(1)\data_label\t_v_dataset_train.pt',
#     r'E:\BaiduNetdiskDownload\duorenwuxuexi(1)\data_label\t_v_dataset_train.pt.bak'
# )

# 保存修改后的数据
with open(r'E:\BaiduNetdiskDownload\duorenwuxuexi(1)\data_label\t_v_dataset_train.pt', 'wb') as f:
    pickle.dump(modified_data, f)