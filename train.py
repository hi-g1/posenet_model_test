import argparse
from train_class import Train
from test_class import Test
from utils import boolean_argument


def main(args, evaluate):

    if not evaluate:
        Train(args).train()
    else:
        Test(args).test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='num of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--img_size', type=int, default=[480,360], help='[width, height]')

    parser.add_argument('--decaying_use', type=boolean_argument, default=True, help='Decaying yes or no')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Decayin_no, sgd 1e-3, adam은 더 느리게')
    parser.add_argument('--initial_learning_rate', type=float, default=0.001, help='Decayin_yes')
    
    parser.add_argument('--device', type=str, default='cuda', help='cpu or gpu')
    parser.add_argument('--wandb_use', type=boolean_argument, default=True, help='log data')
    parser.add_argument('--is_evaluate', type=boolean_argument, default=True, help='True=> Test, False=> Train')
    parser.add_argument('--save_epoch_period', type=int, default=10, help='save period')
    parser.add_argument('--model', type=str, default='vgg16', help='mobilenet, resnet50, vgg16')
    parser.add_argument('--param', type=int, default=64, help='for each model parameter ex) 128, 64')
    parser.add_argument('--beta', type=int, default=250, help='for compute loss in posenet')
    parser.add_argument('--use_default_weight', type=boolean_argument, default=False, help='Yes:Default No:ImageNetV1')

    parser.add_argument('--model_save_path', type=str, default='checkpoint', help='save model path')
    parser.add_argument('--dataset_path', type=str, default='/home/spilab/ws/Cambridge/Street', help='')
    parser.add_argument('--test_model_path', type=str, default='/home/spilab/ws/PoseNet/result/', help='')
    parser.add_argument('--test_dataset_path', type=str, default='/home/spilab/ws/Cambridge/Street', help='')
    
    args, rest_args = parser.parse_known_args()

    main(args, args.is_evaluate)