import math
import os
import pickle
import sys
from random import random
import random

import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import f1_score
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

from ctq_mtl import swin_transformer_s_t_ada

def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    a_v = model.visual_weight
    a_t = (1 - model.visual_weight)
    accu_num1 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_num2 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_num3 = torch.zeros(1).to(device)   # 累计预测正确的样本数

    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8,label0,label1,label2= data
        sample_num += t_1.shape[0]
        # label = label.unsqueeze(1)


        # label = label.unsqueeze(1)


        with torch.no_grad():
            # 4
            pred1,pred2,pred3 = model(v_1.to(device), v_2.to(device), v_3.to(device),
                          v_4.to(device), v_5.to(device), v_6.to(device), v_7.to(device),
                          v_8.to(device), t_1.to(device), t_2.to(device), t_3.to(device),
                          t_4.to(device), t_5.to(device), t_6.to(device), t_7.to(device),
                          t_8.to(device))
        pred1 = pred1.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred1, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num1 += torch.eq(predicted_labels, label0.to(device)).sum()
        loss1 = loss_function(pred1, label0.to(device).long())

        pred2 = pred2.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred2, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num2 += torch.eq(predicted_labels, label1.to(device)).sum()
        loss2 = loss_function(pred2, label1.to(device).long())

        pred3 = pred3.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred3, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num3 += torch.eq(predicted_labels, label2.to(device)).sum()
        loss3 = loss_function(pred3, label2.to(device).long())

        loss = loss1 / (2 * model.w1 * model.w1) + loss2 / (2 * model.w2 * model.w2) + loss3 / (
                    2 * model.w3 * model.w3) + torch.log(model.w1 * model.w2 * model.w3)
        accu_loss += loss


        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc1: {:.3f}, acc2: {:.3f}, acc3: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num1.item() / sample_num,
                                                                               accu_num2.item() / sample_num,
                                                                               accu_num3.item() / sample_num)
        torch.cuda.empty_cache()
    return a_v,a_t
#创建多模态数据集
def evaluate_no_v_2(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    accu_num1 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_num2 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_num3 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    a_v = model.visual_weight
    a_t = (1 - model.visual_weight)
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8,label0,label1,label2= data
        sample_num += t_1.shape[0]
        # label = label.unsqueeze(1)

        noise = lambda x: x + torch.normal(mean=0, std=0.2, size=x.size()).to(device)
        v_1 = noise(v_1.to(device))
        v_2 = noise(v_2.to(device))
        v_3 = noise(v_3.to(device))
        v_4 = noise(v_4.to(device))
        v_5 = noise(v_5.to(device))
        v_6 = noise(v_6.to(device))
        v_7 = noise(v_7.to(device))
        v_8 = noise(v_8.to(device))
        # label = label.unsqueeze(1)


        with torch.no_grad():
            # 4
            pred1,pred2,pred3 = model(v_1.to(device), v_2.to(device), v_3.to(device),
                          v_4.to(device), v_5.to(device), v_6.to(device), v_7.to(device),
                          v_8.to(device), t_1.to(device), t_2.to(device), t_3.to(device),
                          t_4.to(device), t_5.to(device), t_6.to(device), t_7.to(device),
                          t_8.to(device))
        pred1 = pred1.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred1, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num1 += torch.eq(predicted_labels, label0.to(device)).sum()
        loss1 = loss_function(pred1, label0.to(device).long())

        pred2 = pred2.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred2, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num2 += torch.eq(predicted_labels, label1.to(device)).sum()
        loss2 = loss_function(pred2, label1.to(device).long())

        pred3 = pred3.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred3, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num3 += torch.eq(predicted_labels, label2.to(device)).sum()
        loss3 = loss_function(pred3, label2.to(device).long())

        loss = loss1 / (2 * model.w1 * model.w1) + loss2 / (2 * model.w2 * model.w2) + loss3 / (
                    2 * model.w3 * model.w3) + torch.log(model.w1 * model.w2 * model.w3)
        accu_loss += loss


        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc1: {:.3f}, acc2: {:.3f}, acc3: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num1.item() / sample_num,
                                                                               accu_num2.item() / sample_num,
                                                                               accu_num3.item() / sample_num)
        torch.cuda.empty_cache()
    return a_v,a_t,accu_num1.item() / sample_num,accu_num2.item() / sample_num,accu_num3.item()/ sample_num
def evaluate_no_v_5(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    accu_num1 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_num2 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_num3 = torch.zeros(1).to(device)   # 累计预测正确的样本数

    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8,label0,label1,label2= data
        sample_num += t_1.shape[0]
        # label = label.unsqueeze(1)

        noise = lambda x: x + torch.normal(mean=0, std=0.5, size=x.size()).to(device)
        v_1 = noise(v_1.to(device))
        v_2 = noise(v_2.to(device))
        v_3 = noise(v_3.to(device))
        v_4 = noise(v_4.to(device))
        v_5 = noise(v_5.to(device))
        v_6 = noise(v_6.to(device))
        v_7 = noise(v_7.to(device))
        v_8 = noise(v_8.to(device))
        # label = label.unsqueeze(1)


        with torch.no_grad():
            # 4
            pred1,pred2,pred3 = model(v_1.to(device), v_2.to(device), v_3.to(device),
                          v_4.to(device), v_5.to(device), v_6.to(device), v_7.to(device),
                          v_8.to(device), t_1.to(device), t_2.to(device), t_3.to(device),
                          t_4.to(device), t_5.to(device), t_6.to(device), t_7.to(device),
                          t_8.to(device))
        pred1 = pred1.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred1, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num1 += torch.eq(predicted_labels, label0.to(device)).sum()
        loss1 = loss_function(pred1, label0.to(device).long())

        pred2 = pred2.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred2, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num2 += torch.eq(predicted_labels, label1.to(device)).sum()
        loss2 = loss_function(pred2, label1.to(device).long())

        pred3 = pred3.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred3, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num3 += torch.eq(predicted_labels, label2.to(device)).sum()
        loss3 = loss_function(pred3, label2.to(device).long())

        loss = loss1 / (2 * model.w1 * model.w1) + loss2 / (2 * model.w2 * model.w2) + loss3 / (
                    2 * model.w3 * model.w3) + torch.log(model.w1 * model.w2 * model.w3)
        accu_loss += loss


        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc1: {:.3f}, acc2: {:.3f}, acc3: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num1.item() / sample_num,
                                                                               accu_num2.item() / sample_num,
                                                                               accu_num3.item() / sample_num)
        torch.cuda.empty_cache()
    return accu_num1.item() / sample_num,accu_num2.item() / sample_num,accu_num3.item()/ sample_num
def evaluate_no_t_2(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    accu_num1 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_num2 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_num3 = torch.zeros(1).to(device)   # 累计预测正确的样本数

    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8,label0,label1,label2= data
        sample_num += t_1.shape[0]
        # label = label.unsqueeze(1)

        noise = lambda x: x + torch.normal(mean=0, std=0.2, size=x.size()).to(device)
        t_1 = noise(t_1.to(device))
        t_2 = noise(t_2.to(device))
        t_3 = noise(t_3.to(device))
        t_4 = noise(t_4.to(device))
        t_5 = noise(t_5.to(device))
        t_6 = noise(t_6.to(device))
        t_7 = noise(t_7.to(device))
        t_8 = noise(t_8.to(device))
        # label = label.unsqueeze(1)


        with torch.no_grad():
            # 4
            pred1,pred2,pred3 = model(v_1.to(device), v_2.to(device), v_3.to(device),
                          v_4.to(device), v_5.to(device), v_6.to(device), v_7.to(device),
                          v_8.to(device), t_1.to(device), t_2.to(device), t_3.to(device),
                          t_4.to(device), t_5.to(device), t_6.to(device), t_7.to(device),
                          t_8.to(device))
        pred1 = pred1.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred1, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num1 += torch.eq(predicted_labels, label0.to(device)).sum()
        loss1 = loss_function(pred1, label0.to(device).long())

        pred2 = pred2.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred2, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num2 += torch.eq(predicted_labels, label1.to(device)).sum()
        loss2 = loss_function(pred2, label1.to(device).long())

        pred3 = pred3.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred3, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num3 += torch.eq(predicted_labels, label2.to(device)).sum()
        loss3 = loss_function(pred3, label2.to(device).long())

        loss = loss1 / (2 * model.w1 * model.w1) + loss2 / (2 * model.w2 * model.w2) + loss3 / (
                    2 * model.w3 * model.w3) + torch.log(model.w1 * model.w2 * model.w3)
        accu_loss += loss


        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc1: {:.3f}, acc2: {:.3f}, acc3: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num1.item() / sample_num,
                                                                               accu_num2.item() / sample_num,
                                                                               accu_num3.item() / sample_num)
        torch.cuda.empty_cache()
    return accu_num1.item() / sample_num,accu_num2.item() / sample_num,accu_num3.item()/ sample_num
def evaluate_no_t_5(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    accu_num1 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_num2 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_num3 = torch.zeros(1).to(device)   # 累计预测正确的样本数

    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8,label0,label1,label2= data
        sample_num += t_1.shape[0]
        # label = label.unsqueeze(1)

        noise = lambda x: x + torch.normal(mean=0, std=0.5, size=x.size()).to(device)
        t_1 = noise(t_1.to(device))
        t_2 = noise(t_2.to(device))
        t_3 = noise(t_3.to(device))
        t_4 = noise(t_4.to(device))
        t_5 = noise(t_5.to(device))
        t_6 = noise(t_6.to(device))
        t_7 = noise(t_7.to(device))
        t_8 = noise(t_8.to(device))
        # label = label.unsqueeze(1)


        with torch.no_grad():
            # 4
            pred1,pred2,pred3 = model(v_1.to(device), v_2.to(device), v_3.to(device),
                          v_4.to(device), v_5.to(device), v_6.to(device), v_7.to(device),
                          v_8.to(device), t_1.to(device), t_2.to(device), t_3.to(device),
                          t_4.to(device), t_5.to(device), t_6.to(device), t_7.to(device),
                          t_8.to(device))
        pred1 = pred1.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred1, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num1 += torch.eq(predicted_labels, label0.to(device)).sum()
        loss1 = loss_function(pred1, label0.to(device).long())

        pred2 = pred2.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred2, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num2 += torch.eq(predicted_labels, label1.to(device)).sum()
        loss2 = loss_function(pred2, label1.to(device).long())

        pred3 = pred3.squeeze(1)  # 32*11
        pred_classes = F.softmax(pred3, dim=1)
        predicted_labels = torch.argmax(pred_classes, dim=1)
        accu_num3 += torch.eq(predicted_labels, label2.to(device)).sum()
        loss3 = loss_function(pred3, label2.to(device).long())

        loss = loss1 / (2 * model.w1 * model.w1) + loss2 / (2 * model.w2 * model.w2) + loss3 / (
                    2 * model.w3 * model.w3) + torch.log(model.w1 * model.w2 * model.w3)
        accu_loss += loss


        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc1: {:.3f}, acc2: {:.3f}, acc3: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num1.item() / sample_num,
                                                                               accu_num2.item() / sample_num,
                                                                               accu_num3.item() / sample_num)
        torch.cuda.empty_cache()
    return accu_num1.item() / sample_num,accu_num2.item() / sample_num,accu_num3.item()/ sample_num


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


if __name__ == '__main__':
    if torch.cuda.is_available():
        # device = torch.device("cuda")  # 使用默认的GPU设备
        device = torch.device("cuda:0")  # 使用名称为"GeForce_RTXTi"的GPU

    else:
        device = torch.device("cpu")
    batch_size = 8


    tb_writer = SummaryWriter("ctq_mtl_test")


    with open('../../data_label/t_v_dataset_train.pt', 'rb') as file:
        train_dataset = pickle.load(file)

    # 步骤2：从Pickle文件加载测试数据集
    with open('../../data_label/t_v_dataset_test.pt', 'rb') as file:
        test_dataset = pickle.load(file)
    merged_data_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    merged_data_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


    #以上为读取数据集

    #读取模型
    #创建优化器
    epoch_num=25
    # optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=5E-5)
    # #创建优化器
    # epoch_num=25
    #读取训练所需的超参数
    for epoch in range(epoch_num):
        # train
        print(device)
        # print(torch.cuda.get_device_name(1))  # 输出第二块GPU的名称（如果有的话）
        # model = swin_transformer_s_t_ada()

        model = torch.load(f'ctq_mtl_{epoch}.pth')
        model = model.to(device)

        # torch.cuda.empty_cache()


        a_v,a_t,v_2_1,v_2_2,v_2_3=evaluate_no_v_2(model=model,
                                     data_loader=merged_data_test_loader,
                                     device=device,
                                     epoch=epoch)
        v_5_1, v_5_2, v_5_3 = evaluate_no_v_5(model=model,
                                              data_loader=merged_data_test_loader,
                                              device=device,
                                              epoch=epoch)
        t_2_1, t_2_2, t_2_3 = evaluate_no_t_2(model=model,
                                              data_loader=merged_data_test_loader,
                                              device=device,
                                              epoch=epoch)
        t_5_1, t_5_2, t_5_3 = evaluate_no_t_5(model=model,
                                              data_loader=merged_data_test_loader,
                                              device=device,
                                              epoch=epoch)
        tags = ["a_v","a_t","v_2_1","v_2_2","v_2_3","v_5_1","v_5_2","v_5_3","t_2_1","t_2_2","t_2_3","t_5_1","t_5_2","t_5_3"]
        tb_writer.add_scalar(tags[0], a_v, epoch)
        tb_writer.add_scalar(tags[1], a_t, epoch)
        tb_writer.add_scalar(tags[2], v_2_1, epoch)
        tb_writer.add_scalar(tags[3], v_2_2, epoch)
        tb_writer.add_scalar(tags[4], v_2_3, epoch)
        tb_writer.add_scalar(tags[5], v_5_1, epoch)
        tb_writer.add_scalar(tags[6], v_5_2, epoch)
        tb_writer.add_scalar(tags[7], v_5_3, epoch)
        tb_writer.add_scalar(tags[8], t_2_1, epoch)
        tb_writer.add_scalar(tags[9], t_2_2, epoch)
        tb_writer.add_scalar(tags[10], t_2_3, epoch)
        tb_writer.add_scalar(tags[11], t_5_1, epoch)
        tb_writer.add_scalar(tags[12], t_5_2, epoch)
        tb_writer.add_scalar(tags[13], t_5_3, epoch)
