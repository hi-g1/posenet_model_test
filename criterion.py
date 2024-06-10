import torch
import torch.nn as nn
from torch.nn import functional as F

def compute_pose_loss(pred_tr, pred_rot, target_tr, target_rot, beta=300.0):

    pred_rot = F.normalize(pred_rot, p=2, dim=1)
    target_rot = F.normalize(target_rot, p=2, dim=1)

    tr_loss = F.mse_loss(pred_tr, target_tr)
    rot_loss = F.mse_loss(pred_rot, target_rot)
    
    loss = tr_loss + beta * rot_loss

    return loss, tr_loss, rot_loss