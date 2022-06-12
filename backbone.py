from torchvision.models.vgg import vgg11,vgg13,vgg16,vgg19
from torchvision.models.resnet import resnet18,resnet34,resnet50
import torch
import torch.nn as nn


class Backbone_vgg(nn.Module):
    def __init__(self,backbone):
        super(Backbone_vgg, self).__init__()

        if backbone == 'vgg16':
            self.net = vgg16(pretrained=True)
            del self.net.avgpool
            del self.net.classifier

        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))


    def forward(self,x):
        x1 = self.net.features[0:9](x)                      
        x2 = self.net.features[9:16](x1)
        x3 = self.net.features[16:23](x2)
        x4 = self.net.features[23:-1](x3)                 
        return x1,x2,x3,x4


class Backbone_resnet(nn.Module):
    def __init__(self, backbone):
        super(Backbone_resnet, self).__init__()

        if backbone == 'resnet18':
            self.net = resnet18(pretrained=True)

            del self.net.avgpool
            del self.net.fc

        elif backbone == 'resnet34':
            self.net = resnet34(pretrained=True)
            del self.net.avgpool
            del self.net.fc

        elif backbone == 'resnet50':
            self.net = resnet50(pretrained=True)
            del self.net.avgpool
            del self.net.fc

        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def forward(self, x):
        x1 = self.net.conv1(x) 

        x = self.net.bn1(x1)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x2 = self.net.layer1(x)
        x3 = self.net.layer2(x2)
        x4 = self.net.layer3(x3)

        x5 = self.net.layer4(x4)

        return x1, x2, x3, x4, x5

if __name__ == '__main__':
    x = torch.rand(1,3,512,512)
    model = Backbone_resnet(backbone='resnet50')
    print(model)
    x1, x2, x3, x4, x5 = model(x)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)
    print(x5.shape)




















