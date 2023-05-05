import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets ,transforms
from torch import optim

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #定义卷积神经网络模型
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(epoch, criterion, optimizer, model, train_data, test_data, lossp, accuracyp):
    for e in range(1, epoch+1):
        tloss = 0
        #训练一个批次,image是图像，labels是标签
        for images, labels in train_data:
            #前向传播
            output = model.forward(images)
            loss = criterion(output,labels)
            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #记录损失
            tloss += loss.item()
        print("Train epoch %d:\nTrain Loss: %.6f"%(e, tloss/len(train_data)))
        lossp.append(tloss/len(train_data))
        test(model,test_data, accuracyp)
    return model

def test(model,test_data, accuracyp):
    #计算分类的准确率
    correct = 0
    with torch.no_grad():
        for images, lables in test_data:
            output = model.forward(images)
            _, pred = output.max(1)
            num_correct = (pred == lables).sum().item()
            correct += num_correct/images.shape[0]
    print("Test_set Accuracy: %.6f"%(correct/len(test_data)))
    accuracyp.append(correct/len(test_data))

def init():
    model = Net()
    #定义转换器
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5))])
    #下载数据集
    train_set = datasets.mnist.MNIST('./data', train=True, transform=transform, download=True) 
    test_set = datasets.mnist.MNIST('./data', train=False, transform=transform, download=True)
    #使用 pytorch 自带的 DataLoader 定义一个数据迭代器
    params = {
        "epoch" : 20,
        "model" : model,
        "train_data" : DataLoader(train_set,batch_size = 64),
        "test_data" : DataLoader(test_set,batch_size = 64),
        "criterion" : nn.NLLLoss(),
        "optimizer" : optim.SGD(model.parameters(), lr=0.1)
    }
    return params

def draw_plt(loss, accuracy, num):
    plt.title('ConvolutionalNeuralNet Training')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Loss & Accuracy')
    plt.plot(range(1,num+1), loss, 'r', lw=1.5, label='loss')
    plt.plot(range(1,num+1), accuracy, 'b', lw=1.5, label='accuracy')
    plt.legend(loc=0)
    plt.show()

def main():
    params = init()
    loss, accuracy = [], []
    model = train(**params, lossp=loss, accuracyp=accuracy)
    draw_plt(loss, accuracy, params["epoch"])
    torch.save(model.state_dict(), "mnist_2.pt")

if __name__ == "__main__":
    main()