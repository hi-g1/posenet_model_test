import torch
from importlib import reload

import criterion
reload(criterion)

pred_tr = torch.randn(10, 3)
pred_rot = torch.randn(10, 4)

target_tr = torch.randn(10, 3)
target_rot = torch.randn(10, 4)

loss, tr_loss, rot_loss = criterion.compute_pose_loss(pred_tr, pred_rot, target_tr, target_rot)
