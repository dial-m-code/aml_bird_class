import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=200):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class LargeCNN(nn.Module):
    def __init__(self, num_classes=200):
        super(LargeCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.25)

        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(0.25)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.drop_fc2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.drop3(self.pool3(x))

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.drop4(self.pool4(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.drop_fc2(x)
        
        x = self.fc3(x)
        return x

class LargeCNN_MT(nn.Module):
    def __init__(self, num_classes=200, num_attributes=312):
        super(LargeCNN_MT, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.25)

        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(0.25)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.drop_fc2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, num_classes)

        # Classifier attributes
        self.fc1_a = nn.Linear(512, 512)
        self.bn_fc1_a = nn.BatchNorm1d(512)
        self.drop_fc1_a = nn.Dropout(0.5)
        
        self.fc2_a = nn.Linear(512, num_attributes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.drop3(self.pool3(x))

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.drop4(self.pool4(x))

        features = self.avgpool(x)

        # classes
        classes = torch.flatten(features, 1)
        
        classes = F.relu(self.bn_fc1(self.fc1(classes)))
        classes = self.drop_fc1(classes)
        
        classes = F.relu(self.bn_fc2(self.fc2(classes)))
        classes = self.drop_fc2(classes)
        
        classes = self.fc3(classes)

        # attributes
        attributes = torch.flatten(features, 1)
        
        #attributes = F.relu(self.bn_fc1_a(self.fc1_a(attributes)))
        attributes = self.drop_fc1_a(attributes)
        
        attributes = self.fc2_a(attributes)
        
        return classes, attributes

class MediumCNN_MT(nn.Module):
    def __init__(self, num_classes=200, num_attributes=312):
        super(MediumCNN_MT, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # shared layers
        self.fc_shared = nn.Linear(256, 256)
        self.bn_shared = nn.BatchNorm1d(256)
        self.drop_shared = nn.Dropout(0.5)

        # class layer
        self.fc_cls = nn.Linear(256, num_classes)
        
        # attribute layer
        self.fc_attr = nn.Linear(256, num_attributes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)

        features = self.avgpool(x)
        features = torch.flatten(features, 1)

        # shared layers
        features = self.fc_shared(features)
        features = self.bn_shared(features)
        features = F.relu(features)
        features = self.drop_shared(features)

        # class layer
        classes = self.fc_cls(features)

        # attribute layer
        attributes = self.fc_attr(features)
        
        return classes, attributes