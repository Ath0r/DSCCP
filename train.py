import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
os.environ['CUDA_LAUNCH_BLOCKING'] ='1'
import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# from networks.Unet_strip_se import UNet
# from networks.Unet import UNet
from model.DSCAMP_Reformed_UNet import UNet
# from networks.vit_seg_modeling import VisionTransformer as ViT_seg
# from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="/home/yukunkang/DSConv-Net/nets/BRAUnet/BRAU-Netplusplus/synapse_train_test/Synapse/train_npz",
    help="root dir for train data",
)
parser.add_argument(
    "--test_path",
    type=str,
    default="/home/yukunkang/DSConv-Net/nets/BRAUnet/BRAU-Netplusplus/synapse_train_test/Synapse/test_vol_h5",
    help="root dir for test data",
)

parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir")
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
parser.add_argument("--output_dir", type=str, default="./save_models", help="output dir")
parser.add_argument("--max_iterations", type=int, default=90000, help="maximum epoch number to train")
parser.add_argument("--max_epochs", type=int, default=400, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=12, help="batch_size per gpu")
parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
parser.add_argument("--eval_interval", type=int, default=20, help="eval_interval")
parser.add_argument("--model_name", type=str, default="synapse", help="model_name")
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether to use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network base learning rate")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--z_spacing", type=int, default=1, help="z_spacing")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="no: no cache, "
    "full: cache all data, "
    "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
)
parser.add_argument("--resume", help="resume from checkpoint")
parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
)
parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--throughput", action="store_true", help="Test throughput only")
parser.add_argument(
    "--module", help="The module that you want to load as the network, e.g. networks.DAEFormer.DAEFormer"
)

args = parser.parse_args()



if __name__ == "__main__":
    # setting device on GPU if available, else CPU
    # transformer = locate(args.module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()
    # TranUnet 配置
    # cuda = True
    # num_classes = 9
    # backbone = "resnet50"
    # pretrained = True
    # model_path = ''
    # input_shape = (224,224)
    # init_epoch = 0
    # freeze_epoch = 0
    # unFreeze_epoch = 40
    # freeze_batch_size = 0
    # unfreeze_batch_size = 2
    # freeze_train = False
    # init_lr = 1e-4
    # min_lr = init_lr * 0.01
    # optimizer_type = "adam"
    # momentum = 0.9
    # weight_decay = 0
    # lr_decay_type = 'cos'
    # eval_flag = True
    # eval_period = 1
    # num_workers = 2
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # save_dir = 'results'
    # data_path = 'data'
    # image_path = '/home/yukunkang/Data/chest_img'
    # pretrain_path = '/home/yukunkang/DSConv-Net/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    # loss_fuc = "BCEloss"
    # vit_name = 'R50-ViT-B_16'
    # n_skip = 3
    # vit_patches_size = 16



    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        "Synapse": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": 9,
        },
    }
    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.root_path = dataset_config[dataset_name]["root_path"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    net = UNet(in_channels=1, out_channels=9).cuda(0)

    # config_vit = CONFIGS_ViT_seg[vit_name]
    # config_vit.n_classes = num_classes
    # config_vit.n_skip = n_skip
    # if vit_name.find('R50') != -1:
    #     config_vit.patches.grid = (int(input_shape[0] / vit_patches_size), int(input_shape[0] / vit_patches_size))

    # net = ViT_seg(config_vit, img_size=input_shape[0], num_classes=num_classes).to(device)

    trainer = {
        "Synapse": trainer_synapse,
    }
    trainer[dataset_name](args, net, args.output_dir)
