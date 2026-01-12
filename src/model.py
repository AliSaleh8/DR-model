import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self,in_features,out_features,stride=1,downsample=None):

        super(Bottleneck,self).__init__()

        self.conv1=nn.Conv2d(in_features,out_features,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_features)


        self.conv2=nn.Conv2d(out_features,out_features,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_features)

        self.conv3=nn.Conv2d(out_features,out_features*self.expansion,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_features*self.expansion)

        self.downsample=downsample
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        identity=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        if self.downsample is not None:
            identity=self.downsample(x)

        out+=identity
        out=self.relu(out)

        return out

class resnet50(nn.Module):

    def __init__(self,block,layers,num_classes=1000):
        super(resnet50,self).__init__()

        self.in_features=64
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=self._make_layer(64,block,layers[0],stride=1)
        self.layer2=self._make_layer(128,block,layers[1],stride=2)
        self.layer3=self._make_layer(256,block,layers[2],stride=2)
        self.layer4=self._make_layer(512,block,layers[3],stride=2)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*block.expansion,num_classes)


    def _make_layer(self,out_features,block,nb_blocks,stride=1):
        downsample=None

        if stride!=1 or self.in_features!=out_features* block.expansion:
            downsample=nn.Sequential(
                nn.Conv2d(self.in_features,out_features*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_features*block.expansion)
            )

        layer=[]
        layer.append(block(self.in_features,out_features,stride=stride,downsample=downsample))
        self.in_features=out_features*block.expansion

        for _ in range(1,nb_blocks):
            layer.append(block(self.in_features,out_features))


        return nn.Sequential(*layer)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.fc(x)

        return x

