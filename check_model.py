import os
from importlib import reload
from PIL import Image

import torch
import torchvision

import models.posenet
reload(models.posenet)

m = models.posenet.PoseNet()

x = torch.randn(1, 3, 360, 480)

tr,rot = m(x)