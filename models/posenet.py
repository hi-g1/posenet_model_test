import torch
import torchvision

class PoseNet(torch.nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()

        mobilenet = torchvision.models.mobilenet_v2(
                weights=torchvision.models.MobileNet_V2_Weights.DEFAULT) 

        self.backbone = torch.nn.Sequential(*list(mobilenet.children())[:-1])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))

        self.trans_head = torch.nn.Sequential(
                torch.nn.Linear(1280 * 7 * 7, 1024), 
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 3) )
        
        self.rot_head = torch.nn.Sequential(
                torch.nn.Linear(1280 * 7 * 7, 1024), 
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 4) )

    def forward(self, x):
        
        x = self.backbone(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        trans = self.trans_head(x)
        rot = self.rot_head(x)

        return  trans, rot