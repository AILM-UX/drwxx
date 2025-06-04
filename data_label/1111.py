import pickle
import os

file_path = r'E:\BaiduNetdiskDownload\duorenwuxuexi(1)\data_label\t_v_dataset_train.pt'

try:
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    print("文件加载成功")
except EOFError:
    print("文件为空或损坏，请重新生成文件")
except Exception as e:
    print(f"加载失败: {e}")