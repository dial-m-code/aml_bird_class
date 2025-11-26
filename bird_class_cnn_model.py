import torch.nn as nn

class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        
        self.conv_layer2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_layer3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        
        self.conv_layer4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same")
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  
        flattened_features = 256 * 4 * 4
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(flattened_features, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.relu_fc = nn.ReLU()
        
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv_layer1(x)))
        out = self.relu2(self.bn2(self.conv_layer2(out)))
        out = self.max_pool1(out)

        out = self.relu3(self.bn3(self.conv_layer3(out)))
        out = self.relu4(self.bn4(self.conv_layer4(out)))
        out = self.max_pool2(out)
                
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.bn_fc(out)
        out = self.relu_fc(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out