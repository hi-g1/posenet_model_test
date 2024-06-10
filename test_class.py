import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import functional as F
import models.posenet
from utils import AverageMeter , position_dist, rotation_dist
import cambridge
from tqdm import tqdm
import os

class Test(object):
    def __init__(self, args):
        self.args = args
        self.image_width = (args.img_size)[0]
        self.image_height = (args.img_size)[1]
        self.batch_size = args.batch_size
        self.device = args.device
        self.model = args.model
        self.param = 0
        self.test_model_path = args.test_model_path
        self.test_dataset_path = args.test_dataset_path

        # cuda or cuda
        if self.device == "cuda" and torch.cuda.is_available():
            device = torch.device('cuda')
        else: device = torch.device('cpu')
        print(f"✅ [using {device} device]")

    # 이미지 전처리 함수 정의
    def preprocess_image(self, image_path):
        
        # basic transform
        transform = transforms.Compose([
            transforms.Resize((self.image_height, self.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])

        # load dataset
        dataset_root = self.test_dataset_path
        test_set = cambridge.CambridgeDataset(dataset_root, 'test', transform=transform)

        return test_set
        
    def test(self):

        folder_name = []
        model_name = []
        model_header = []
        model_path = []

        # result 폴더 안에 있는 폴더 목록 가져오기
        subfolders = [f.path for f in os.scandir(self.test_model_path) if f.is_dir()]

        for subfolder in subfolders:
            # name_model_param_bool_beta_time 형식의 폴더인지 확인
            # print(subfolder.split("/")[-1])
            # subfolder =  subfolder.split("/")[-1]
            
            name, model, param, bool_info, beta, time_info = subfolder.split("/")[-1].split("_")

            # best.pth 파일 경로 생성
            best_pth_path = os.path.join(subfolder, "best.pth")

            # 파일이 존재하는지 확인 후 출력
            if os.path.exists(best_pth_path):
                # print("Folder Name:", subfolder)
                # print("Model:", model)
                # print("Param:", param)
                # print("Beta:", beta)
                # print("Best.pth Path:", best_pth_path)
                folder_name.append(subfolder)
                model_name.append(model)
                model_header.append(param)
                model_path.append(best_pth_path)
        
        print('✅ [get subfolder results]\n')

        for option_index in range(len(model_name)):

            self.model = model_name[option_index]
            self.param = int(model_header[option_index])
            print(f'✅ [set folder name with {folder_name[option_index].split("/")[-1]}]')
            print(f'✅ [set model name with {self.model}]')
            print(f'✅ [set model param with {self.param}]\n')

            # 모델 설정하기
            if self.model == "mobilenet":
                m = models.posenet.PoseMobile(self.param, self.args.use_default_weight)
            elif self.model == "resnet50":
                m = models.posenet.PoseRes(self.param, self.args.use_default_weight)
            elif self.model == "vgg16":
                m = models.posenet.PoseVgg(self.param, self.args.use_default_weight)

            model = m.to(self.device)

            # 저장된 가중치 불러오기
            # for name, param in model.named_parameters():
            #     print(f"{name}: {param.size()}")

            if os.path.exists(model_path[option_index]):
                model.load_state_dict(torch.load(model_path[option_index]), strict=False)
            
            model.eval()  # 모델을 evaluation mode로 설정

            # 이미지 전처리
            input_image = self.preprocess_image(self.test_dataset_path)
            # print(input_image)

            error_tr_meter = AverageMeter()
            error_rot_meter = AverageMeter()

            model.eval()
            test_set = self.preprocess_image(self.test_dataset_path)
            for idx, (image, target_tr, target_rot) in tqdm(enumerate(test_set) ): 

                image = image.to(self.device)

                target_tr = target_tr.to(self.device)

                target_rot = target_rot.to(self.device)
                target_rot = F.normalize(target_rot, p=2, dim=0)

                with torch.no_grad():
                    pred_tr, pred_rot = model(image.unsqueeze(0)) 

                pred_tr = pred_tr.squeeze(0)

                pred_rot = pred_rot.squeeze(0)
                pred_rot = F.normalize(pred_rot, p=2, dim=0)

                pred_rot = pred_rot.cpu().detach().numpy()
                target_rot = target_rot.cpu().detach().numpy()

                pred_tr = pred_tr.cpu().detach().numpy()
                target_tr = target_tr.cpu().detach().numpy()

                #    pred_rot = utils.quat_to_euler(pred_rot)
                #    target_rot = utils.quat_to_euler(target_rot)

                error_tr = position_dist(pred_tr, target_tr)
                error_rot = rotation_dist(pred_rot, target_rot)

                error_tr_meter.update(error_tr)
                error_rot_meter.update(error_rot)

            name = folder_name[option_index].split("/")[-1]
            print(f"{name} error_tr: {error_tr_meter.avg}")
            print(f"{name} error_rot: {error_rot_meter.avg}\n")

            error_tr_meter.reset()
            error_rot_meter.reset()