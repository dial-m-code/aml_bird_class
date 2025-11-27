import torch
import torch.nn as nn

class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        
        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        
        self.conv_layer4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same")
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.conv_layer5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same")
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        
        self.max_pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout3 = nn.Dropout(0.3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(128, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.relu_fc = nn.ReLU()
        self.dropout_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv_layer1(x)))
        out = self.relu2(self.bn2(self.conv_layer2(out)))
        out = self.max_pool1(out)
        out = self.dropout1(out)

        out = self.relu3(self.bn3(self.conv_layer3(out)))
        out = self.relu4(self.bn4(self.conv_layer4(out)))
        out = self.max_pool2(out)
        out = self.dropout2(out)
        
        out = self.relu5(self.bn5(self.conv_layer5(out)))
        out = self.max_pool3(out)
        out = self.dropout3(out)
                
        out = self.avgpool(out)
        out = self.flatten(out)

        out = self.fc1(out)
        out = self.bn_fc(out)
        out = self.relu_fc(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        
        return out