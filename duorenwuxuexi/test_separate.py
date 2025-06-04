import math
import os
import pickle
import sys
from random import random
import random

import pandas as pd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm

from MTFromer import swin_transformer_s_t_ada

#创建多模态数据集
class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks):
        super(MultiTaskLoss, self).__init__()
        self.num_tasks = num_tasks
        self.loss_functions = [nn.CrossEntropyLoss() for _ in range(num_tasks)]  # 为每个任务选择适当的损失函数

    def forward(self, outputs, targets):
        total_loss = 0.0
        for i in range(self.num_tasks):
            task_loss = self.loss_functions[i](outputs[i], targets[i])
            total_loss += task_loss
        return total_loss
def train_one_epoch(model, optimizer, data_loader, device, epoch):


    model.train()
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num1 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_num2 = torch.zeros(1).to(device)   # 累计预测正确的样本数

    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        t_1,t_2,t_3,t_4,t_5,t_6,t_7,t_8,v_1,v_2,v_3,v_4,v_5,v_6,v_7,v_8,label0,label1,label2,label3,label4,label5 = data
        sample_num += t_1.shape[0]
        # label = label.unsqueeze(1)

        pred1= model(v_1.to(device), v_2.to(device), v_3.to(device),v_4.to(device),v_5.to(device), v_6.to(device), v_7.to(device),
                                    v_8.to(device),t_1.to(device), t_2.to(device), t_3.to(device),
                                    t_4.to(device),t_5.to(device), t_6.to(device), t_7.to(device),
                                    t_8.to(device))
        pred1 = pred1.squeeze(1)#32*11
        # print(pred1.shape)
        pred_classes = F.softmax(pred1, dim=1)
        # print("pred_classes",pred_classes)
        predicted_labels = torch.argmax(pred_classes, dim=1)

        # print("predicted_labels",predicted_labels)
        # print(pred1)
        # print(pred1.shape)
        accu_num1 += torch.eq(predicted_labels, label0.to(device)).sum()
        loss1 = loss_function(pred1, label0.to(device).long())


        loss=loss1
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc1: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num1.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num1.item() / sample_num

def evaluate(model, data_loader, device,  epoch=1):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num1 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_num2 = torch.zeros(1).to(device)   # 累计预测正确的样本数

    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8,label0,label1,label2,label3,label4,label5 = data
        sample_num += t_1.shape[0]
        # label = label.unsqueeze(1)


        # label = label.unsqueeze(1)


        with torch.no_grad():
            # 4
            pred1 = model(v_1.to(device), v_2.to(device), v_3.to(device),
                          v_4.to(device), v_5.to(device), v_6.to(device), v_7.to(device),
                          v_8.to(device), t_1.to(device), t_2.to(device), t_3.to(device),
                          t_4.to(device), t_5.to(device), t_6.to(device), t_7.to(device),
                          t_8.to(device))
        pred1 = pred1.squeeze(1)  # 32*11
        # print(pred1.shape)
        pred_classes = F.softmax(pred1, dim=1)
        # print("pred_classes",pred_classes)
        predicted_labels = torch.argmax(pred_classes, dim=1)

        # print("predicted_labels",predicted_labels)
        # print(pred1)
        # print(pred1.shape)
        accu_num1 += torch.eq(predicted_labels, label1.to(device)).sum()
        loss1 = loss_function(pred1, label1.to(device).long())


        loss = loss1
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc1: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num1.item() / sample_num
                                                                               )
        torch.cuda.empty_cache()
    return accu_loss.item() / (step + 1), accu_num1.item() / sample_num


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
        label3= label[3]
        label3 = torch.tensor(label3)
        label4 = label[4]
        label4 = torch.tensor(label4)
        label5 = label[5]
        label5 = torch.tensor(label5)
        # 在此处进行任何必要的数据预处理，例如将图像转换为张量
        return image_t1,image_t2,image_t3,image_t4,image_t5,image_t6,image_t7,image_t8, image_v1,image_v2,image_v3,image_v4,image_v5,image_v6,image_v7,image_v8,label0,label1,label2,label3,label4,label5


# 创建自定义数据集实例


if __name__ == '__main__':

    batch_size = 8


    # 步骤2：从Pickle文件加载测试数据集
    with open('../data_label/t_v_dataset_test.pt', 'rb') as file:
        test_dataset = pickle.load(file)

    merged_data_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    if torch.cuda.is_available():
        # device = torch.device("cuda")  # 使用默认的GPU设备
        device = torch.device("cuda:0")  # 使用名称为"GeForce_RTXTi"的GPU
    else:
        device = torch.device("cpu")
    #以上为读取数据集
    model = torch.load('separate_1.pth')

    epoch_num=1


        # train
        # print(torch.cuda.get_device_name(1))  # 输出第二块GPU的名称（如果有的话）
    val_loss, val_acc1 = evaluate(model=model,data_loader=merged_data_test_loader,device=device)

    print(val_loss, val_acc1)