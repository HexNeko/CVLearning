#coding=utf-8

#数据集图片为划分测试集和数据集
#这部分程序用于把每一类的图片随机划分为2:3两部分，分别作为测试集和训练集

import os
import shutil
import random

orgin_path = r"./data/15-Scene"
goal_path = r"./data/minitest"

val_path = os.path.join(goal_path, "val")
train_path = os.path.join(goal_path, "train")


#分类
for type in os.listdir(orgin_path):
    #每一类
    print("type:", type)
    full_dir = os.path.join(orgin_path, type)
    img = [img_name for img_name in os.listdir(full_dir)]
    print("data size:", len(img))
    #随机
    random.shuffle(img)
    #划分测试集和训练集
    divide_index = int(len(img)/5*2)
    val = img[0:divide_index]
    train = img[divide_index:]
    print("val size: %d, train size: %d"%(len(val), len(train)))
    #如果没有路径就创建文件夹
    for path in [val_path, train_path]:
        if not os.path.isdir(os.path.join(path, type)):
            os.makedirs(os.path.join(path, type))
    #文件复制
    for v in val:
        shutil.copy(os.path.join(full_dir, v), os.path.join(val_path, type + '/' + v))
    for t in train:
        shutil.copy(os.path.join(full_dir, t), os.path.join(train_path , type + '/' + t))
    