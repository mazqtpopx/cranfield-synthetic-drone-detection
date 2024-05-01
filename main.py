import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# from datasets import DroneDataset
from datasets import DroneDataset_Base
from datasets import DroneDataset_DroneOnly
# from datasets import ValidationDataset
from datasets_jpeg_comp import DroneDataset_Base_JPEG


import torch
import json


import matplotlib.pyplot as plt
import pickle
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp

# import wandb


from engine import train_one_epoch, evaluate
import utils
import transforms as T
import os

import argparse

from config import SYNTH_MASKED_DATASET, SYNTH_MASKED_VALIDATION_DATASET, MODEL_PATH, DISTRIBUTED, NUM_WORKERS
from config import EPOCHS, BATCH_SIZE, SYNTH_MASKED_ADV_DATASET_1, SYNTH_MASKED_ADV_DATASET_2, SYNTH_MASKED_ADV_DATASET_3, SYNTH_MASKED_DATASET_2, SYNTH_MASKED_DATASET_3
from config import SINGLE_DRONE_DATASET_MAVIC, SINGLE_DRONE_DATASET_FPV, SINGLE_DRONE_DATASET_INSPIRE
from config import MULTI_DRONE_50M, MULTI_DRONE_100M, MULTI_DRONE_200M, MULTI_DRONE_400M, MULTI_DRONE_800M
from config import MULTI_DRONE_BIRDS, DISTRACTORS_GENERIC, DISTRACTORS_REALISTIC, PSYCHEDELIC_BACKGROUNDS

from evaluate_lib import evaluate_on_real_drone_datasets

from config import MAV_VID_COCO_PATH, DRONEBIRD_COCO_PATH, ANTI_UAV_COCO_PATH
from config import MAV_VID_DATASET_NAME, DRONEBIRD_DATASET_NAME, ANTI_UAV_DATASET_NAME
from config import MAV_VID_IMG_DIR, DRONEBIRD_IMG_DIR, ANTI_UAV_IMG_DIR
from config import LEARNING_RATE

from config import NOISE_ENABLED, JPEG_COMPRESSION_ENABLED

import random 
import imgaug.augmenters as iaa

COCO_PATHS = [MAV_VID_COCO_PATH, DRONEBIRD_COCO_PATH, ANTI_UAV_COCO_PATH]
DATASET_NAMES = [MAV_VID_DATASET_NAME, DRONEBIRD_DATASET_NAME, ANTI_UAV_DATASET_NAME]
IMG_DIRS = [MAV_VID_IMG_DIR, DRONEBIRD_IMG_DIR, ANTI_UAV_IMG_DIR]

# from coco_metrics import get_coco_metrics

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def run_fn(fn, world_size, learning_rate, test_name, test_group, dataset_name, jpeg_compression, noise, augmentations, dataset_size):
    #FOR WINDOWS, USE SPAWN. FOR LINUX, USE FORK
    #From: https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
    #BUT!!!! for CUDA, spawn should be fine.... 
    mp.spawn(fn, args=(world_size, learning_rate, test_name, test_group, dataset_name, jpeg_compression, noise, augmentations, dataset_size), nprocs=world_size, join=True)

    
def cleanup():
    dist.destroy_process_group()


def get_fasterrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model



#transforms...
#So I moved the train transforms to get_transform_train
def get_transform(train):
    print("Getting transforms...")
    transforms = []
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if train:
        print("train enabled")
        transforms.append(T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ))
        
        transforms.append(T.ToTensor())
        transforms.append(T.RandomPhotometricDistort())

        print("finished setting up transforms")
    else:
        transforms.append(T.ToTensor())
    print("returning transforms")
    return T.Compose(transforms)




def main(rank, world_size, learning_rate, test_name, test_group, dataset_name, jpeg_compression, noise, augmentations, dataset_size = None):

    print(f"running on {rank=} and {world_size=}")
    torch.distributed.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size = world_size)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.cuda.set_device(rank)

    print(f"{device=}")


    print("defining dataset")
    # our dataset has 4 classes - background, drone, propeller, & camera
    num_classes = 2

    #multi_drones
    dataset_multidrone_50m = DroneDataset_Base_JPEG(MULTI_DRONE_50M, get_transform(train=augmentations), jpeg_compression, noise)
    dataset_multidrone_100m = DroneDataset_Base_JPEG(MULTI_DRONE_100M, get_transform(train=augmentations), jpeg_compression, noise, dataset_size)
    dataset_multidrone_200m = DroneDataset_Base_JPEG(MULTI_DRONE_200M, get_transform(train=augmentations), jpeg_compression, noise)
    dataset_multidrone_400m = DroneDataset_Base_JPEG(MULTI_DRONE_400M, get_transform(train=augmentations), jpeg_compression, noise)
    dataset_multidrone_800m = DroneDataset_Base_JPEG(MULTI_DRONE_800M, get_transform(train=augmentations), jpeg_compression, noise)

    dataset_multidrone_birds = DroneDataset_Base_JPEG(MULTI_DRONE_BIRDS, get_transform(train=augmentations), jpeg_compression, noise)
    dataset_distractors_generic = DroneDataset_Base_JPEG(DISTRACTORS_GENERIC, get_transform(train=augmentations), jpeg_compression, noise)
    dataset_distractors_realistic = DroneDataset_Base_JPEG(DISTRACTORS_REALISTIC, get_transform(train=augmentations), jpeg_compression, noise)
    dataset_psychedelic_backgrounds = DroneDataset_Base_JPEG(PSYCHEDELIC_BACKGROUNDS, get_transform(train=augmentations), jpeg_compression, noise)
    if dataset_name == "multi_drone_100m":
        dataset = dataset_multidrone_100m
    elif dataset_name == "multi_drone_200m":
        dataset = dataset_multidrone_200m
    elif dataset_name == "multi_drone_400m":
        dataset = dataset_multidrone_400m
    elif dataset_name == "multi_drone_800m":
        dataset = dataset_multidrone_800m
    elif dataset_name == "multi_drone_50m":
        dataset = dataset_multidrone_50m
    elif dataset_name == "multi_drone_birds":
        dataset = dataset_multidrone_birds
    elif dataset_name == "distractors_generic":
        dataset = dataset_distractors_generic
    elif dataset_name == "distractors_realistic":
        dataset = dataset_distractors_realistic
    elif dataset_name == "psychedelic_backgrounds":
        dataset = dataset_psychedelic_backgrounds
    elif dataset_name == "multi_drone_combined":
        dataset = torch.utils.data.dataset.ConcatDataset([dataset_multidrone_50m, dataset_multidrone_100m, dataset_multidrone_200m])
    elif dataset_name == "distractors_combined":
        dataset = torch.utils.data.dataset.ConcatDataset([dataset_multidrone_birds, dataset_distractors_generic, dataset_distractors_realistic])
    else:
        raise Exception("No dataset selected. Please select a dataset using -dataset <dataset-name>. --help for more info")

    print("Finished defining dataset")


    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    if DISTRIBUTED:
        print("Distributed enabled. Loading samplers")
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler,
                                                            BATCH_SIZE, drop_last=True)

        print("Starting data loaders")
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_batch_sampler,
            num_workers=4, collate_fn = utils.collate_fn,
            pin_memory = True
        )
        print("Finished data_loader")

    print("Finished loading samplers")


    # get the model using our helper function
    model = get_fasterrcnn_model(num_classes)

    print("moving model to device")
    # move model to the right device
    model.to(rank)

    model_without_ddp = model
    if DISTRIBUTED: 
        print(rank)
        type(rank)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[rank],
                                                          output_device=rank)
        model_without_ddp = model.module
        print("finished moving model")

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate,
                                momentum=0.9, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler()


    num_epochs = EPOCHS

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=3,
                                                  gamma=0.1)


    loss_vals = []
    mavvid_ap_vals = []
    dvb_ap_vals = []
    antiuav_ap_vals = []
    epochs = []
    last_epoch = 0

    for epoch in range(num_epochs):
        print(f"training epoch {epoch}")
        epochs.append(epoch)
        if DISTRIBUTED:
            train_sampler.set_epoch(epoch)
            print(f"Epoch set")
            # train for one epoch, printing every 10 iterations

        print(f"training one epoch")
        met_logger, loss = train_one_epoch(model, optimizer, data_loader, rank, epoch, 10)
        loss_detached = loss.detach().cpu().numpy().item()

        loss_vals.append(loss_detached)
        print(f"training on the second dataset")
        lr_scheduler.step()

        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            #"args": args,
            "epoch": epoch,
        }
        print("saving checkpoint...")
        #/home/observation/repos/drone_model_detector/model/multi_drone/{TEST_GROUP}/{TEST_NAME} exist. if not make it 
        if not os.path.exists(f"/home/observation/repos/drone_model_detector/model/multi_drone/{test_group}/{test_name}"):
            os.makedirs(f"/home/observation/repos/drone_model_detector/model/multi_drone/{test_group}/{test_name}")
        utils.save_on_master(checkpoint, f"/home/observation/repos/drone_model_detector/model/multi_drone/{test_group}/{test_name}/lr_{learning_rate}_epoch_{epoch}.pt")

        last_epoch = epoch
  
    model.eval()
    evaluate_on_real_drone_datasets(model, device, test_name, test_group, learning_rate, last_epoch)

    print("That's it!")
    cleanup()


if __name__ == '__main__':
    #parse args
    parser = argparse.ArgumentParser(description='Process input arguments')

    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('-test_name', '--test_name', type=str, default="test_name")
    parser.add_argument('-test_group', '--test_group', type=str, default="test_group")
    parser.add_argument('-dataset', '--dataset', type=str, default="multi_drone_100m")
    #0 for disabled, 1 for enabled


    parser.add_argument('--jpeg_compression', action='store_true')
    parser.add_argument('--no-jpeg_compression', dest='jpeg_compression', action='store_false')
    parser.set_defaults(jpeg_compression=JPEG_COMPRESSION_ENABLED)

    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--no-noise', dest='noise', action='store_false')
    parser.set_defaults(noise=NOISE_ENABLED)

    
    parser.add_argument('--augmentations', action='store_true')
    parser.add_argument('--no-augmentations', dest='augmentations', action='store_false')
    parser.set_defaults(augmentations=True)

    parser.add_argument('-dataset_size', '--datset_size', type=int, default=5000)

    args = parser.parse_args()

    if not os.path.exists(f"training_logging/{args.test_group}/{args.test_name}"):
        os.makedirs(f"training_logging/{args.test_group}/{args.test_name}")

    #write a txt file to the location containing all the args
    with open(f"training_logging/{args.test_group}/{args.test_name}/training_args.txt", 'w') as f:
        f.write(f"{args.learning_rate=}\n")
        f.write(f"{args.test_name=}\n")
        f.write(f"{args.test_group=}\n")
        f.write(f"{args.dataset=}\n")
        f.write(f"{args.jpeg_compression=}\n")
        f.write(f"{args.noise=}\n")
        f.write(f"{args.augmentations=}\n")
        f.write(f"{args.datset_size=}\n")
        f.write(f"{EPOCHS=}\n")
        f.write(f"{BATCH_SIZE=}\n")


    #torch.multiprocessing.freeze_support()
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("starting program")
    print(f"args: {args.learning_rate=}, {args.test_name=}, {args.test_group=}, {args.dataset=}, {args.jpeg_compression=}, {args.noise=}, {args.augmentations=}, {args.datset_size=}")
    print("finished setting up distributed env")
    n_gpus = torch.cuda.device_count()
    print(f"Number of gpus available: {n_gpus}")

    run_fn(main, n_gpus, args.learning_rate, args.test_name, args.test_group, args.dataset, args.jpeg_compression, args.noise, args.augmentations, args.datset_size)