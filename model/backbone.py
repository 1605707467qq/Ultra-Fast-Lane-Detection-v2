import torch,pdb
import torchvision
import torch.nn.modules

class vgg16bn(torch.nn.Module):
    def __init__(self,pretrained = False):
        super(vgg16bn,self).__init__()
        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        model = model[:33]+model[34:43]
        self.model = torch.nn.Sequential(*model)
        
    def forward(self,x):
        return self.model(x)
class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        elif layers == '34fca':
            model = torch.hub.load('cfzd/FcaNet', 'fca34' ,pretrained=True)
        else:
            raise NotImplementedError
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2,x3,x4

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights,efficientnet_v2_s,EfficientNet_V2_S_Weights
class Efficientnet(torch.nn.Module):
    '''
    代表efficient系列并不是指定模型
    输入无要求
    输出1280维度
    '''
    def __init__(self,layers, pretrained=False):
        super(Efficientnet,self).__init__()
        # mb 7.7GB显存
        if layers=='efb0':
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1) # b0 输出维度为1280
        elif layers=='efv2s':
            self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1) # v2_s 输出维度为1280
        self.features = self.model.features
        self.avgpool = self.model.avgpool
        self.classifier = self.model.classifier
    def forward(self,x):
        x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.squeeze(x, dim=-1)
        # x = torch.squeeze(x, dim=-1)
        return x
from torchvision.models import regnet_y_1_6gf, RegNet_Y_1_6GF_Weights,regnet_y_3_2gf,RegNet_Y_3_2GF_Weights
class Regnet(torch.nn.Module):
    '''
    输入无要求
    输出7392维度
    '''
    def __init__(self,layers, pretrained=False):
        super(Regnet,self).__init__()
        # 2.4gb
        # RegNet_Y_1_6GF_Weights.IMAGENET1K_V2 # 输出888维度
        # 43.2mb
        if layers == 'regy1_6':
            model = regnet_y_1_6gf(weights=RegNet_Y_1_6GF_Weights.IMAGENET1K_V2) # 1.6GF 输出888维度
        elif layers == 'regy3_2':
            model = regnet_y_3_2gf(weights=RegNet_Y_3_2GF_Weights.IMAGENET1K_V2) # 3.2GH 输出1512维度
        self.stem = model.stem
        self.trunk_output = model.trunk_output
        self.avgpool = model.avgpool
    def forward(self,x):
        x = self.stem(x)
        # with torch.no_grad():
        x = self.trunk_output(x)
        x = self.avgpool(x)
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
    
        return x