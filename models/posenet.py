import torch
import torchvision

class PoseMobile(torch.nn.Module):
    def __init__(self, param, use_default_weight):
        super(PoseMobile, self).__init__()
        
        if use_default_weight:
            mobilenet = torchvision.models.mobilenet_v2(\
                weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
        else:
            mobilenet = torchvision.models.mobilenet_v2(\
                weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2) 

        self.backbone = torch.nn.Sequential(*list(mobilenet.children())[:-1])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))

        self.trans_head = torch.nn.Sequential(
                torch.nn.Linear(1280 * 7 * 7, param), 
                torch.nn.ReLU(),
                torch.nn.Linear(param, 3) )
        
        self.rot_head = torch.nn.Sequential(
                torch.nn.Linear(1280 * 7 * 7, param), 
                torch.nn.ReLU(),
                torch.nn.Linear(param, 4) )

    def forward(self, x):
        
        x = self.backbone(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        trans = self.trans_head(x)
        rot = self.rot_head(x)

        return  trans, rot
    
class PoseRes(torch.nn.Module):
    def __init__(self, param, use_default_weight):
        super(PoseRes, self).__init__()
        
        if use_default_weight:
            resnet50 = torchvision.models.resnet50(\
                weights= torchvision.models.ResNet50_Weights.DEFAULT)
        else:
            resnet50 = torchvision.models.resnet50(\
                weights= torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

        self.backbone = torch.nn.Sequential(*list(resnet50.children())[:-1])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))

        self.trans_head = torch.nn.Sequential(
                torch.nn.Linear(2048 * 7 * 7, param), 
                torch.nn.ReLU(),
                torch.nn.Linear(param, 3) )
        
        self.rot_head = torch.nn.Sequential(
                torch.nn.Linear(2048 * 7 * 7, param), 
                torch.nn.ReLU(),
                torch.nn.Linear(param, 4) )

    def forward(self, x):
        
        x = self.backbone(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        trans = self.trans_head(x)
        rot = self.rot_head(x)

        return  trans, rot
    
class PoseVgg(torch.nn.Module):
    def __init__(self, param, use_default_weight):
        super(PoseVgg, self).__init__()

        if use_default_weight:
            vgg16 = torchvision.models.vgg16(weights= torchvision.models.VGG16_Weights.DEFAULT)
        else:
            vgg16 = torchvision.models.vgg16(weights= torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES)

        self.backbone = torch.nn.Sequential(*list(vgg16.children())[:-1])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))

        self.trans_head = torch.nn.Sequential(
                torch.nn.Linear(512 * 7 * 7, param), 
                torch.nn.ReLU(),
                torch.nn.Linear(param, 3) )
        
        self.rot_head = torch.nn.Sequential(
                torch.nn.Linear(512 * 7 * 7, param), 
                torch.nn.ReLU(),
                torch.nn.Linear(param, 4) )

    def forward(self, x):
        
        x = self.backbone(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        trans = self.trans_head(x)
        rot = self.rot_head(x)

        return  trans, rot
