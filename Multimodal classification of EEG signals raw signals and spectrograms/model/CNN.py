import torch.nn as nn
import torch

channels_num = 64
time_len=1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
order=4
dim=16
maxpoor_kerel=12

class CNN(nn.Module):
    def __init__(self, num_features=128):
        super(CNN, self).__init__()
        self.num_features = num_features

        self.BN1 = nn.BatchNorm2d(1)
        self.BN2 = nn.BatchNorm1d(10)
        self.cnn=nn.Conv2d(1,5,kernel_size=(64,17))
        self.avgpool=nn.AvgPool2d(kernel_size=(1,112))

        #self.GRU = nn.GRU(input_size=3*dim, hidden_size=8, batch_first=True)
        self.relu=nn.ReLU()

        self.FC = torch.nn.Sequential(nn.Dropout(0.5), nn.Linear(5, 5),nn.Dropout(0.5),
                                      nn.Linear(5, 2), nn.Softmax())

        self.dropout = nn.Dropout(0.2)

        self.flatten = nn.Flatten()
        self.to(device)

    def forward(self, data):  # data

        x= data
        x = torch.transpose(x, 2, 1)
        # x = torch.reshape(x, (32, channels_num, int(128*time_len)))  # 32，64，128，self.num_features
        x = torch.unsqueeze(x, dim=1)  # 32，1,64，128
        x = self.BN1(x)
        x=self.cnn(x)
        x = self.relu(x)
        x=self.avgpool(x)


        x = self.flatten(x)
        x = self.FC(x)
        return x, []


