import os

#Replace this with your (e.g. /home/{user}/ on linux)
REPOS_DIR = "/home/observation"

#Path to the faster rcnn model (.pt)
MODEL_PATH = f"{REPOS_DIR}/repos/drone_model_detector/model/multi_drone/100m_tests/lr_0_00002.pt"
OUTPUT_DIR = f"{REPOS_DIR}/repos/drone_model_detector/model/"

#Directory containing the datasets
DATASET_DIR = f"{REPOS_DIR}/repos/datasets/"

#Directory names containing the datasets
ANTIUAV_DIR = os.path.join(DATASET_DIR, "anti-uav")
DRONEBIRD_DATASET = os.path.join(DATASET_DIR, "drone-vs-bird")
MAVVID_DATASET = os.path.join(DATASET_DIR, "mav-vid")


MAV_VID_DATASET_NAME = "mav-vid"
DRONEBIRD_DATASET_NAME = "drone-vs-bird"
ANTI_UAV_DATASET_NAME = "anti-uav"

MAV_VID_COCO_PATH = "dataset_coco_files/mav-vid/val.json"
DRONEBIRD_COCO_PATH = "dataset_coco_files/drone-vs-bird/val.json"
ANTI_UAV_COCO_PATH = "dataset_coco_files/anti-uav/val-rgb.json"

MAV_VID_IMG_DIR = f"{REPOS_DIR}/repos/datasets/mav-vid/imgs"
DRONEBIRD_IMG_DIR = f"{REPOS_DIR}/repos/datasets/drone-vs-bird/imgs"
ANTI_UAV_IMG_DIR = f"{REPOS_DIR}/repos/datasets/anti-uav/"

COCO_PATHS = [MAV_VID_COCO_PATH, DRONEBIRD_COCO_PATH, ANTI_UAV_COCO_PATH]
DATASET_NAMES = [MAV_VID_DATASET_NAME, DRONEBIRD_DATASET_NAME, ANTI_UAV_DATASET_NAME]
IMG_DIRS = [MAV_VID_IMG_DIR, DRONEBIRD_IMG_DIR, ANTI_UAV_IMG_DIR]




SYNTH_MASKED_DATASET = os.path.join(DATASET_DIR, "synth_masked_drones_big")
SYNTH_MASKED_VALIDATION_DATASET = os.path.join(DATASET_DIR, "synth_masked_drones")
SYNTH_MASKED_DATASET_2 = os.path.join(DATASET_DIR, "drone_dataset_less_birds_bigger_bounds")
SYNTH_MASKED_DATASET_3 = os.path.join(DATASET_DIR, "drone_dataset_style_augmentation")
SYNTH_MASKED_ADV_DATASET_1 = os.path.join(DATASET_DIR, "drone_dataset_adversarial_background")
SYNTH_MASKED_ADV_DATASET_2 = os.path.join(DATASET_DIR, "drone_dataset_adversarial_background_postprocess")
SYNTH_MASKED_ADV_DATASET_3 = os.path.join(DATASET_DIR, "drone_dataset_adversarial_background_postprocess_v2")

SINGLE_DRONE_DATASET_MAVIC = os.path.join(DATASET_DIR, "single_drones/single_drone_mavic")
SINGLE_DRONE_DATASET_FPV = os.path.join(DATASET_DIR, "single_drones/single_drone_fpv")
SINGLE_DRONE_DATASET_INSPIRE = os.path.join(DATASET_DIR, "single_drones/single_drone_inspire")

MULTI_DRONE_50M = os.path.join(DATASET_DIR, "multi_drones/multi_drone_no_birds_50m")
MULTI_DRONE_100M = os.path.join(DATASET_DIR, "multi_drones/multi_drone_no_birds_100m")
MULTI_DRONE_200M = os.path.join(DATASET_DIR, "multi_drones/multi_drone_no_birds_200m")
MULTI_DRONE_400M = os.path.join(DATASET_DIR, "multi_drones/multi_drone_no_birds_400m")
MULTI_DRONE_800M = os.path.join(DATASET_DIR, "multi_drones/multi_drone_no_birds_800m")


MULTI_DRONE_BIRDS = os.path.join(DATASET_DIR, "multi_drones/drones_with_birds_100m")
DISTRACTORS_GENERIC = os.path.join(DATASET_DIR, "multi_drones/drones_generic_distractors_100m")

DISTRACTORS_REALISTIC = os.path.join(DATASET_DIR, "multi_drones/drones_realistic_distractors_100m")

PSYCHEDELIC_BACKGROUNDS = os.path.join(DATASET_DIR, "multi_drones/psychedelic_backgrounds")

#Set true if running on multiple GPUs
DISTRIBUTED = True



#Paramerters for the training
NUM_WORKERS = 4

#number of machines
WORLD_SIZE = 4
#number of gpus
RANK = [0, 1, 2, 3]


#HYPERPARAMETERS
LEARNING_RATE = 0.00002
EPOCHS = 10
BATCH_SIZE = 8

#NB: below values are DEFAULTS, they will be overwritten by argparse if specified
NOISE_ENABLED = False
JPEG_COMPRESSION_ENABLED = False

MIN_DRONE_AREA = 1

