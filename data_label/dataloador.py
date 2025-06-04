import os
import pickle
import random

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms


class vit_tac_Dataset(Dataset):
    def __init__(self, dataset_dir,label_dir,window_size=8,transform_t=None,transform_v=None):
        self.dataset_dir = dataset_dir
        self.labels_df = pd.read_excel(label_dir)
        self.window_size=window_size
        self.data_t,self.data_v= self.load_data()
        self.transform_t=transform_t
        self.transform_v=transform_v

    def load_data(self):
        data_t = []
        data_v=[]
        for label_folder in os.listdir(self.dataset_dir):
            target_row = self.labels_df[self.labels_df.iloc[:, 0] == label_folder]  # 通过 .iloc 方法按位置访问第一列
            if not target_row.empty:
                labels = target_row.iloc[0, 1:].values
                label_folder_path = os.path.join(self.dataset_dir, label_folder)

                # 检查是否是文件夹
                if os.path.isdir(label_folder_path):
                    for v_or_t_folder in os.listdir(label_folder_path):
                        if v_or_t_folder == "tactile":
                            sub_folder_path = os.path.join(label_folder_path, v_or_t_folder)

                            # 检查是否是文件夹
                            if os.path.isdir(sub_folder_path):
                                image_files = [f for f in os.listdir(sub_folder_path) if
                                               f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                                for i in range(0, len(image_files) - self.window_size - 6, 8):
                                    data_t.append((sub_folder_path, image_files[i:i + self.window_size], labels))
                        else:
                            sub_folder_path = os.path.join(label_folder_path, v_or_t_folder)

                            # 检查是否是文件夹
                            if os.path.isdir(sub_folder_path):

                                image_files = [f for f in os.listdir(sub_folder_path) if
                                               f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                                for i in range(0, len(image_files) - self.window_size - 6, 8):
                                    data_v.append((sub_folder_path, image_files[i:i + self.window_size], labels))
        return data_t,data_v

    def __len__(self):
        return len(self.data_t)

    def __getitem__(self, idx):

        images_t = []
        images_v = []

        folder_path, image_names, label = self.data_t[idx]
        for image_name in image_names:
            image_path = os.path.join(folder_path, image_name)
            image = Image.open(image_path).convert("RGB")

            if self.transform_t:
                image = self.transform_t(image)

            images_t.append(image)

        folder_path, image_names, label = self.data_v[idx]
        for image_name in image_names:
            image_path = os.path.join(folder_path, image_name)
            image = Image.open(image_path).convert("RGB")

            if self.transform_v:
                image = self.transform_v(image)

            images_v.append(image)
        image_t1,image_t2,image_t3,image_t4,image_t5,image_t6,image_t7,image_t8=images_t
        image_v1,image_v2,image_v3,image_v4,image_v5,image_v6,image_v7,image_v8=images_v
        label0=label[0]
        label0 = torch.tensor(label0)  # 将标签转换为PyTorch张量
        label1 = label[1]
        label1 = torch.tensor(label1)
        label2 = label[2]
        label2 = torch.tensor(label2)
        # 在此处进行任何必要的数据预处理，例如将图像转换为张量
        return image_t1,image_t2,image_t3,image_t4,image_t5,image_t6,image_t7,image_t8, image_v1,image_v2,image_v3,image_v4,image_v5,image_v6,image_v7,image_v8,label0,label1,label2


# 创建自定义数据集实例

transform_t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(size=112,scale=(0.8,1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_v = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(size=224,scale=(0.8,1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
transform_t1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(size=112,scale=(0.8,1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_v1 = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomResizedCrop(size=224,scale=(0.8,1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


if __name__ == '__main__':
    total_dataset_dir = 'E:/duorenwuxuexi(1)/dataset/tvl_dataset/hct/data'
    label_dir = 'label_final_output.xlsx'
    total_dataset = vit_tac_Dataset(total_dataset_dir,label_dir, window_size=8, transform_t=transform_t,transform_v=transform_v)
    random.seed(10)
    total_indices = list(range(len(total_dataset)))
    random.shuffle(total_indices)
    train_size = int(0.8 * len(total_indices))  # 80% 划分给训练集
    valid_size = len(total_indices) - train_size  # 剩余部分划分给验证集



    train_indices = total_indices[:train_size]
    train_dataset = Subset(total_dataset, train_indices)
    save_path_train = "t_v_dataset_train2.pt"
    with open(save_path_train, 'wb') as file:
        pickle.dump(train_dataset, file)
    total_dataset = vit_tac_Dataset(total_dataset_dir,label_dir, window_size=8, transform_t=transform_t1,transform_v=transform_v1)
    valid_indices = total_indices[train_size:]
    test_dataset = Subset(total_dataset, valid_indices)
    save_path_test = "t_v_dataset_test2.pt"
    with open(save_path_test, 'wb') as file:
        pickle.dump(test_dataset, file)