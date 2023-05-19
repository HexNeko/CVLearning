from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode

#基于https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py修改

type = int(input("chose mode: 1->vgg16, 2->resnet50\n"))
#vgg16 or resnet50
model_chose = 'resnet50'
if(type == 1):
    model_chose = 'vgg16'
elif(type == 2):
    model_chose = 'resnet50'

save_dir = 'model'
data_dir = './data/test'
#data_dir = './hymenoptera_data/hymenoptera_data'

save_path = ''
#加载数据
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("prediction model:", type)
print("size:", dataset_sizes)
print("class:", class_names)

# Training the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            count = 0
            total = len(dataloaders[phase])
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if(count%50 == 0):
                    print('\ntraining: %d/%d'%(count,total), end='')
                elif(count%5 == 0):
                    print('.', end='')
                count+=1

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Visualizing the model predictions
# Generic function to display predictions for a few images
def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
    

# Finetuning the ConvNet
# Load a pretrained model and reset final fully connected layer.

model_ft = None

if model_chose == "vgg16":
    #use vgg16 model
    # Sequential
    #   (0): Linear(in_features=25088, out_features=4096, bias=True)
    #   (1): ReLU(inplace=True)
    #   (2): Dropout(p=0.5, inplace=False)
    #   (3): Linear(in_features=4096, out_features=4096, bias=True)
    #   (4): ReLU(inplace=True)
    #   (5): Dropout(p=0.5, inplace=False)
    #   (6): Linear(in_features=4096, out_features=1000, bias=True)
    save_path = save_dir + '/vgg16_test.pt'
    model_ft = models.vgg16(weights = 'VGG16_Weights.DEFAULT')
    model_ft.classifier._modules['6'].out_features = len(class_names)
    model_ft = model_ft.to(device)

elif model_chose == "resnet50":
    #use resnet50 model
    save_path = save_dir + '/resnet_test.pt'
    model_ft = models.resnet50(weights = 'ResNet50_Weights.DEFAULT')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

else:
    save_path = save_dir + '/default_test.pt'
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)


criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# Train and evaluate
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)
# save
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
torch.save(model_ft.state_dict(), save_path)

#参考https://blog.csdn.net/weixin_38468077/article/details/121671139
def plot_confusion_matrix(conf_matrix, classes, classes_num, title='Confusion matrix',):
    # 绘制混淆矩阵
    print(conf_matrix)
    # 显示数据
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2	#数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(classes_num):
        for y in range(classes_num):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                    verticalalignment='center',
                    horizontalalignment='center',
                    color="white" if info > thresh else "black")
    plt.title(title)                
    plt.tight_layout()#保证图不重叠
    plt.yticks(range(classes_num), classes)
    plt.xticks(range(classes_num), classes,rotation=45)#X轴字体倾斜45°

def confusion_matrix(preds, classes, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, classes):
        conf_matrix[p, t] += 1

def creat_confusion_matrix(model, test_loader):
    #首先定义一个 分类数*分类数 的空混淆矩阵
    conf_matrix = torch.zeros(len(class_names), len(class_names))
    # 使用torch.no_grad()可以显著降低测试用例的GPU占用
    with torch.no_grad():
        for step, (imgs, targets) in enumerate(test_loader):
            out = model(imgs)
            #记录混淆矩阵参数
            confusion_matrix(out, targets, conf_matrix)
    return conf_matrix

conf_matrix = creat_confusion_matrix(model_ft, dataloaders['val'])
plot_confusion_matrix(conf_matrix, class_names, len(class_names), 'Confusion matrix')
visualize_model(model_ft, 10)


plt.show()
plt.pause(0)
plt.close()

