import os
import torch
from torchvision import transforms
import wandb
import cambridge
from models.posenet import PoseNet
import criterion
from utils import AverageMeter
from datetime import datetime
from pytz import timezone

class Train(object):
    def __init__(self, args):
        
        self.args = args
        self.image_width = (args.img_size)[0]
        self.image_height = (args.img_size)[1]
        self.batch_size = args.batch_size
        self.num_epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.initial_learning_rate = args.initial_learning_rate
        self.device = args.device
        self.model_save_path = args.model_save_path
        self.wandb_use = args.wandb_use
        self.save_epoch_period = args.save_epoch_period
        self.dataset_path = args.dataset_path
        self.best_val_loss = 1e10
        self._train_set = 0
        self._valid_set = 0
        self.start_time = datetime.now(timezone('Asia/Seoul')).strftime("%y%m%d%H%M")
        
        # Pose_modelname_param_lr_beta_time
        self.save_path_with_time = 'Pose_' + '_' + str(self.start_time[2:]) 

        if self.wandb_use:
            wandb.init(project='Training PosNet')
            wandb.run.name = self.save_path_with_time
            wandb.run.save()

            model_args = {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.img_size
            }

            wandb.config.update(model_args)
            print('wandb 사용 중')

        if not os.path.exists(self.save_path_with_time):
            os.makedirs(self.save_path_with_time)
        print("디렉토리 생성 완료")
        
        # cuda or cuda
        if self.device == "cuda" and torch.cuda.is_available():
            device = torch.device('cuda')
        else: device = torch.device('cpu')
        print(f"using device {device}...")

    def data_loader(self):
        # basic transform
        transform = transforms.Compose([
            transforms.Resize((self.image_height, self.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])

        # load dataset
        dataset_root = self.dataset_path
        train_set = cambridge.CambridgeDataset(dataset_root, 'train', transform=transform)

        # split train and val set
        split_ratio = 0.8
        seed = 42
        torch.manual_seed(seed)
        train_set, val_set = torch.utils.data.random_split(
            train_set, \
            [int(len(train_set)*split_ratio), \
            len(train_set)-int(len(train_set)*split_ratio)]
        )
        self._train_set = train_set
        self._valid_set = val_set
        # print(type(train_set))


        train_loader = torch.utils.data.DataLoader(train_set, batch_size= self.batch_size, shuffle=True)
        # print((next(iter(train_loader)))[0].size())
        val_loader = torch.utils.data.DataLoader(val_set, batch_size= self.batch_size, shuffle=True)

        return train_loader, val_loader 
    
    def train(self):

        m = PoseNet()
        model = m.to(self.device)

        # 옵티마이저
        optimizer = torch.optim.SGD(m.parameters(), lr=self.learning_rate, momentum=0.9)

        # 데이터셋 로드
        train_loader, val_loader = self.data_loader()

        # 데이터 시각화
        loss_meter = AverageMeter()
        tr_loss_meter = AverageMeter()
        rot_loss_meter = AverageMeter()


        # Train the model
        # print(total_step)
        # print(f"using {device} device...")
        # print(total_step)

        for epoch in range(self.num_epochs):

            # for param_group in optimizer.param_groups:
            #     print('learing rage: ', param_group['lr'])

            # train
            model.train()
            print ('------------------- Train: Epoch [{}/{}] -------------------'.format(epoch+1, self.num_epochs) )

            loss_meter.reset()
            tr_loss_meter.reset()
            rot_loss_meter.reset()

            for index, _train_set in enumerate(train_loader):
                # print(f"Loss: {loss_meter.avg}")
                image,target_tr,target_rot = _train_set
                image = image.to(self.device)
                target_tr = target_tr.to(self.device)
                target_rot = target_rot.to(self.device)

                # Forward pass
                pred_tr,pred_rot = model(image)

                loss, tr_loss, rot_loss  = criterion.compute_pose_loss(pred_tr, pred_rot, target_tr, target_rot, self.beta)

                # print(f"loss : {loss} loss_item : {image.size()[0]}")

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, error_if_nonfinite = True)
                optimizer.step()

                # logging
                loss_meter.update(loss.item(), image.size()[0] )
                tr_loss_meter.update(tr_loss.item(), image.size()[0] )
                rot_loss_meter.update(rot_loss.item(), image.size()[0] )

            if self.wandb_use:
                wandb.log({"Training loss": loss_meter.avg})
                wandb.log({"Training tr_loss": tr_loss_meter.avg})
                wandb.log({"Training rot_loss": rot_loss_meter.avg})
                # wandb.log({"learning_rate": rot_loss_meter.avg})

            print ('==> Train. loss: {:.4f}  tr_loss: {:.4f}  rot_loss: {:.4f}\n'.format(loss_meter.avg, tr_loss_meter.avg, rot_loss_meter.avg) )

            # val
            model.eval()
            print ('------------------- Val.: Epoch [{}/{}] -------------------'.format(\
                epoch+1, self.num_epochs) )

            loss_meter.reset()
            tr_loss_meter.reset()
            rot_loss_meter.reset()
            
            with torch.no_grad():
                for index, _valid_set in enumerate(val_loader):
                    image,target_tr,target_rot = _valid_set
                    image = image.to(self.device)
                    target_tr = target_tr.to(self.device)
                    target_rot = target_rot.to(self.device)

                    # Forward pass
                    pred_tr,pred_rot = model(image)
                    loss, tr_loss, rot_loss = criterion.compute_pose_loss(pred_tr, pred_rot, target_tr, target_rot, self.beta)
                    # print(loss)

                    # logging
                    loss_meter.update(loss.item(), image.size()[0] )
                    tr_loss_meter.update(tr_loss.item(), image.size()[0] )
                    rot_loss_meter.update(rot_loss.item(), image.size()[0] )

                    # if (index+1) % 10 == 0:
                    #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, tr_loss: {:.4f}  rot_loss: {:.4f}'.format(\
                    #         epoch+1, self.num_epochs, index+1, total_step, loss_meter.avg, tr_loss_meter.avg, rot_loss_meter.avg) )

                if self.wandb_use:
                    wandb.log({"Valid loss": loss_meter.avg})
                    wandb.log({"Valid tr_loss": tr_loss_meter.avg})
                    wandb.log({"Valid rot_loss": rot_loss_meter.avg})

            print ('==> Val. loss: {:.4f}  tr_loss: {:.4f}  rot_loss: {:.4f}'.format(loss_meter.avg, tr_loss_meter.avg, rot_loss_meter.avg) )

            if loss_meter.avg < self.best_val_loss:
                self.best_val_loss = loss_meter.avg
                torch.save(model.state_dict(), self.save_path_with_time + '/best.pth')
                print(f"[best model saved]\n")
            
            if (epoch+1) % self.save_epoch_period == 0:
                torch.save(model.state_dict(), self.save_path_with_time + '/epoch'+ str(epoch+1) +'.pth')
                print(f"[step {epoch+1} model saved]\n")