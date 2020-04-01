import torch.nn as nn
import torch.nn.functional as F
#网络模型LeNet和AlexNet

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()#28*28
        self.conv1 = nn.Conv2d(1, 20, 3, 1)#3*3的卷积核
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))#output:26*26*20
        x = F.max_pool2d(x, 2, 2)#output:13*13*20
        x = F.relu(self.conv2(x))#output:11*11*50
        x = F.max_pool2d(x, 2, 2)#output:5*5*50
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))#output:500
        x = self.fc2(x)#output:10
        return F.log_softmax(x, dim=1)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        # print(self.conv1.weight[0][2],len(self.conv1.weight))

        self.features = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
