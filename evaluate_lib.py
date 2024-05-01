

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

#pytorch
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

#python pkgs
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import json

from lib.iou import compute_iou, convert_to_xyxy

from config import DRONEBIRD_DATASET
from config import COCO_PATHS, DATASET_NAMES, IMG_DIRS
from config import MAV_VID_DATASET_NAME, DRONEBIRD_DATASET_NAME, ANTI_UAV_DATASET_NAME

from coco_metrics import get_coco_metrics

#taken from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def load_model(model_path, device = "cuda:0"):
    #load the machine learning model that we're evaluating
    device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = get_model_instance_segmentation(num_classes)

    model.load_state_dict(torch.load(model_path)['model'])
    model.to(device)

    model.eval()
    return model

def load_faster_rcnn_model(model_path, device = "cuda:0"):
    #load the machine learning model that we're evaluating
    device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = get_fasterrcnn_model(num_classes)

    model.load_state_dict(torch.load(model_path)['model'])
    model.to(device)

    model.eval()
    return model

def get_fasterrcnn_model(num_classes):
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

#now, get the model to predict bbox of drone
def get_img(img_id, coco_path, img_dir):
    
    #coco_path = os.path.join(DRONEBIRD_DATASET, "train.json")
    #img_dir = os.path.join(DRONEBIRD_DATASET, "imgs")
    with open(coco_path) as f:
        coco_json = json.load(f)
        print(coco_json['info'])
        print(coco_json['images'][img_id])
        #load img
        img = cv2.imread(os.path.join(img_dir, coco_json['images'][img_id]['file_name']))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    
def get_ground_truth(img_id, coco_path, img_dir):
#     coco_path = os.path.join(DRONEBIRD_DATASET, "train.json")
#     img_dir = os.path.join(DRONEBIRD_DATASET, "imgs")
    with open(coco_path) as f:
        coco_json = json.load(f)
        print(coco_json['info'])
        # print(coco_json['images'][img_id])
        #load img
        file_name = [x for x in coco_json['images'] if x['id'] == img_id][0]
        print(f"{file_name=}")
        # img = cv2.imread(os.path.join(img_dir, coco_json['images'][img_id]['file_name']))
        #27/06/2023 - fixed so that it works with anti-uav val set - as the IDs don't start with 1, but something like 150000....
        img = cv2.imread(os.path.join(img_dir, file_name['file_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #plt.imshow(img)
        #plt.show()

        #27/06/2023 - removed, becuase img_id is passed to the funct and this one is not needed
        # img_id = coco_json['images'][img_id]['id']

        #load the annotation
        #a = coco_json['annotations'][0]
        #print(f"{a=}")
        #ann = list(filter(lambda x:x['annotations']['img_id'] == img_id, coco_json))
        ann = [x for x in coco_json['annotations'] if x['image_id'] == img_id]
        #ann = coco_json['annotations']
        #ann = coco_json['annotations'][img_id]
        # print(f"{ann=}")
        bbox = ann[0]['bbox']

        # print(f"{bbox=}")
        return bbox
    

#inputs: 
# masks - list of masks to draw
# drawing_frame - frame on which we draw the masks
# colour - (r,g,b) with r,g,b being vals between 0,255
def draw_masks(masks, drawing_frame, colour):

    height, width, channels = drawing_frame.shape
    total_mask = np.zeros((height,width), dtype=bool)
    for mask in masks:
        total_mask += mask
    #NB we have to invert mask because for some reason PIL masks are inverted
    np_mask = np.uint8(~total_mask)*255
    #where we have a 0, convert it to 128 - this will be the are where the drone is, and we want it to be transparent
    np_mask[np_mask == 0] = 128
    pil_mask = Image.fromarray(np_mask)
    # pil_mask.show() # can show for debug


    #generate a green image, half trasnaprent
    colour_img = Image.new('RGBA', (width,height), colour)
    #colour_img.show()
    # transparent_output_img = img_draw.convert('RGBA')
    img_PIL = Image.fromarray(drawing_frame)
    #draw the green img on the original image but only the masked area
    drawing_frame = Image.composite(img_PIL, colour_img, pil_mask).convert('RGB')

    return np.asarray(drawing_frame)

def draw_single_mask(mask, drawing_frame, colour):

    height, width, channels = drawing_frame.shape
    total_mask = np.zeros((height,width), dtype=bool)
#     for mask in masks:
#         total_mask += mask
    total_mask += mask
    #NB we have to invert mask because for some reason PIL masks are inverted
    np_mask = np.uint8(~total_mask)*255
    #where we have a 0, convert it to 128 - this will be the area where the drone is, and we want it to be transparent
    np_mask[np_mask == 0] = 128
    pil_mask = Image.fromarray(np_mask)
    # pil_mask.show() # can show for debug


    #generate a green image, half trasnaprent
    colour_img = Image.new('RGBA', (width,height), colour)
    #colour_img.show()
    # transparent_output_img = img_draw.convert('RGBA')
    img_PIL = Image.fromarray(drawing_frame)
    #draw the green img on the original image but only the masked area
    drawing_frame = Image.composite(img_PIL, colour_img, pil_mask).convert('RGB')

    #convert back to numpy array 
    drawing_frame = np.asarray(drawing_frame)
    return drawing_frame

def process_detected_objects(detected_object):
    filtered_boxes = torchvision.ops.nms(detected_object[0]['boxes'],detected_object[0]['scores'], 0.8)

    # print(filteredBoxes)

    boxes = detected_object[0]['boxes'].cpu().detach().numpy()
    classes = detected_object[0]['labels'].cpu().detach().numpy()
    masks = None
    if 'masks' in detected_object[0]:    
        masks = (detected_object[0]['masks'] > 0.5).squeeze().cpu().detach().numpy()
    # print("boxes detedted")
    # print(boxes)


    scores = detected_object[0]['scores'].cpu().detach().numpy()

    return filtered_boxes, boxes, classes, masks, scores

class DetectedObjects():
    def __init__(self, img_id, filtered_boxes, boxes, classes, masks, scores):
        self.img_id = img_id
        self.filtered_boxes = filtered_boxes
        self.boxes = boxes
        self.classes = classes
        self.masks = masks
        self.scores = scores

    def get_detection(self):
        return self.filtered_boxes, self.boxes, self.classes, self.masks, self.scores

    def __len__(self):
        return len(self.filtered_boxes)

#like process_detected_objects, but for processing outputs from a batch from a model
#outputs the string to append to the json file (instead of raw boxes...)
def process_detected_objects_batch(detected_object, batch_imgs_ids, batch_size):
    detected_objects = []
    for i in range(0,batch_size):
        filtered_boxes = torchvision.ops.nms(detected_object[i]['boxes'],detected_object[i]['scores'], 0.8)
        boxes = detected_object[i]['boxes'].cpu().detach().numpy()
        classes = detected_object[i]['labels'].cpu().detach().numpy()
        masks = None
        if 'masks' in detected_object[i]:
            masks = (detected_object[i]['masks'] > 0.5).squeeze().cpu().detach().numpy()
        scores = detected_object[i]['scores'].cpu().detach().numpy()
        detected_objects.append(DetectedObjects(batch_imgs_ids[i], filtered_boxes, boxes, classes, masks, scores))
    return detected_objects




    
def draw_predicted_boxes_masks(predicted_filtered_boxes, predicted_boxes, predicted_classes, predicted_masks, predicted_scores, ground_truth, drawing_frame):
    #Make these global vars in the future...
    #This is a threshold used to filter out low score predictions
    PREDICTION_THRESHOLD = 0.7

    #Threshold for IoU over which the prediction/GT is countred as a true positive
    IOU_TRHESHOLD = 0.5
    
    #Contains the detected masks (above the prediction_threshold)
    #tp contains true positive masks (drawn in green)
    #fp contains false positive masks (drawn in red)
    tp_masks = []
    fp_masks = []

    #nb: i denotes the index of the detected object - the detected objects are stored in an array
    #in the filtered_boxes, boxes, classes, masks, scores variables, and each index corresponds to the
    #same item
    for i in predicted_filtered_boxes:
        
        #explain why we want the threshold to be above the threshold...
        if predicted_scores[i] > PREDICTION_THRESHOLD:
            
            if (compute_iou(convert_to_xyxy(ground_truth),predicted_boxes[i,:]) >= IOU_TRHESHOLD):
                print(f"Predicted a drone correctly: {i=}")
                
                #true positive, draw green
                cv2.rectangle(drawing_frame, (int(predicted_boxes[i,0]),int(predicted_boxes[i,1])), (int(predicted_boxes[i,2]),int(predicted_boxes[i,3])), (0,255,0), 2)
                # drawing_frame.rectangle(predicted_boxes[i,:],fill=None,outline="rgb(0,255,0)")
                # tp_masks.append(predicted_masks[i])
                
                drawing_frame = draw_single_mask(predicted_masks[i], drawing_frame, (0,255,0))
            elif(compute_iou(convert_to_xyxy(ground_truth),predicted_boxes[i,:]) < IOU_TRHESHOLD):
                print(f"Predicted a drone incorrectly: {i=}")
                print(f"box: {predicted_boxes[i,:]=}")
                #false positive, draw red
                # drawing_frame.rectangle(predicted_boxes[i,:],fill=None,outline="rgb(255,0,0)")
                cv2.rectangle(drawing_frame, (int(predicted_boxes[i,0]),int(predicted_boxes[i,1])), (int(predicted_boxes[i,2]),int(predicted_boxes[i,3])), (255,0,0), 2)
                # fp_masks.append(predicted_masks[i])

    #Draw masks
#     print(f"{predicted_masks.shape=}")
    #Draw all masks 
    # drawing_frame = draw_masks(predicted_masks, drawing_frame, (0,255,0))
    return drawing_frame


def draw_ground_truth(img, ground_truth, colour):
    bbox = ground_truth
    return cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), colour, 2)


def convert_to_PIL_and_push_to_GPU(frame, device="cuda:0"):
    trans = transforms.ToTensor()
    img_PIL = Image.fromarray(frame)
    img_tens = trans(img_PIL).unsqueeze(0)
    input = img_tens.to(device)
    return input

def apply_transforms_batch(batch, device="cuda:0"):
    # trans = transforms.ToTensor()
    # batch_tens = torch.from_numpy(batch).type(torch.FloatTensor)
    #convert from numpy unit8 to float... (0-255 to 0-1)
    batch = batch/255.0
    batch_tens = torch.from_numpy(batch)
    # transforms
    # batch_tens = trans(batch)
    # batch_tens = trans(batch)
    input_batch = batch_tens.to(device, dtype=torch.float)
    # print(f"{input_batch.shape=}")
    return input_batch






def setup_output_file(coco_path):
    output_file = {
        "info":"",
        "licenses":"",
        "images":"",
        "annotations":"",
        "categories":"",
        "segment_info":"",
    }

    #copy the info, licenses, and images
    with open(coco_path) as f:
        coco_json = json.load(f)    
        output_file['info'] = coco_json['info']
        output_file['licenses'] = coco_json['licenses']
        output_file['images'] = coco_json['images']
        output_file['categories'] = coco_json['categories']
    
    return output_file


def convert_xyxy_to_xywh(bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    
    w = x2-x1
    h = y2-y1
    return [x1,y1,w,h]
#input: xywh!!!!!!!
def get_area(bbox):
    return bbox[2]*bbox[3]

def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #we shouldnt resize the images - becuase the bounding boxes will not match the GT!
    # img = cv2.resize(img, (1920,1080))
    img = img.transpose(2,0,1)
    return img




#use this if dataset has different image sizes throughout (e.g. drone-vs-bird)
def process_dataset_without_batches(model, device, coco_path, img_dir, dataset_name, prediction_threshold):
    # model.eval()

    output_file = setup_output_file(coco_path)
    annotations_json = []
    annotation_id = 1
    with open(coco_path) as f:
        coco_json = json.load(f) 
        #batch, channels, height, width
        for image in coco_json['images']:
            # old code - without batching
            img_id = image['id']
            print(f"{img_id=}")
            # print(f"{os.path.join(IMG_DIRS[ds_i], image['file_name'])=}")
            # frame = cv2.imread(os.path.join(IMG_DIRS[ds_i], image['file_name']))
            # frame = load_image(os.path.join(IMG_DIRS[ds_i], image['file_name']))
            img_path = os.path.join(img_dir, image['file_name'])
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input = convert_to_PIL_and_push_to_GPU(frame, device)
            with torch.no_grad():
                detectedObject = model(input)
                filtered_boxes, boxes, classes, masks, scores = process_detected_objects(detectedObject)
                output_BB = []
                for i in filtered_boxes:
                    #filter the boxes based on a threshold 
                    if scores[i] > prediction_threshold:
                        #append to the output_BB list. While we're at it, convert to ints (boxes is in floats at this point)
                        output_BB.append([int(x) for x in boxes[i,:]])

                        bbox_xywh = convert_xyxy_to_xywh([int(x) for x in boxes[i,:]])
                        area = get_area(bbox_xywh)
                        
                        string_to_append = {
                            "id": str(annotation_id),
                            "image_id": img_id,
                            "category_id": 1,                  
                            "bbox": bbox_xywh,
                            "segmentation": [],
                            "area": area,
                            "score": float(scores[i]),
                            "iscrowd": 0
                        }
                        
                        annotations_json.append(string_to_append)
                        annotation_id += 1
                        # print(f"{annotations_json=}")
        # break # break for loop - only do one annotation
    
    output_file['annotations'] = annotations_json

    # with open(f'coco_metrics/{dataset_name}-{model_name}.json', 'w') as f:
    #     output_json = json.dump(output_file, f)
    #         # sys.exit()

    # print(f"Finished generating the coco_eval of {model_name} for {dataset_name} dataset")
    # print(f"Saved results as coco_metrics/{dataset_name}-{model_name}.json")
    
    return output_file

def process_dataset_with_batches(model, device, coco_path, img_dir, dataset_name, prediction_threshold, batch_size):
    output_file = setup_output_file(coco_path)
    annotations_json = []
    annotation_id = 1
    with open(coco_path) as f:
        coco_json = json.load(f) 
        j = 0
        #batch, channels, height, width
        input_batch = np.zeros((batch_size, 3, 1080,1920))
        batch_imgs_ids = []
        for image in coco_json['images']:
            # j = 0 #counter for the batch
            if j == batch_size-1:
                # print(f"{input_batch.shape=}")

                #add the last image to the batch
                img_filename = os.path.join(img_dir, image['file_name'])
                print(f"evaluating img {j}")
                
                img = load_image(img_filename)
                input_batch[j,:,:,:] = img
                # print(f"{input_batch.shape=}")

                batch_imgs_ids.append(image['id'])
                
                #transform to tensor
                input_batch = apply_transforms_batch(input_batch, device=device)
                detectedObjects = model(input_batch)

                # print(f"{batch_imgs_ids=}")
                detected_objects = process_detected_objects_batch(detectedObjects, batch_imgs_ids, batch_size)
                for detected_object in detected_objects:
                    for i in range(0,len(detected_object)):
                        #filter the boxes based on a threshold 
                        if detected_object.scores[i] > prediction_threshold:
                            #append to the output_BB list. While we're at it, convert to ints (boxes is in floats at this point)
                            # output_BB.append([int(x) for x in detected_object.boxes[i,:]])

                            bbox_xywh = convert_xyxy_to_xywh([int(x) for x in detected_object.boxes[i,:]])
                            area = get_area(bbox_xywh)
                            # print(f"{annotation_id=}")
                            print(f"{detected_object.img_id=}")
                            string_to_append = {
                                "id": str(annotation_id),
                                "image_id": detected_object.img_id,
                                "category_id": 1,                  
                                "bbox": bbox_xywh,
                                "segmentation": [],
                                "area": area,
                                "score": float(detected_object.scores[i]),
                                "iscrowd": 0
                            }
                            annotations_json.append(string_to_append)
                            annotation_id += 1
                j = 0   
                batch_imgs_ids = []
                input_batch = np.zeros((batch_size, 3, 1080,1920))
            else:
                #add to the batch#
                img_filename = os.path.join(dataset_name, image['file_name'])
                # import os.path

                img = load_image(img_filename)

                input_batch[j,:,:,:] = img
                # normal_test = img
                # print(f"{input_batch=}")
                batch_imgs_ids.append(image['id'])
                # print(f"{j=}")
                # print(f"{batch_imgs_ids=}")
                j += 1

    output_file['annotations'] = annotations_json

    return output_file


def evaluate_on_real_drone_datasets(model, device, test_name, test_group, learning_rate, epoch):
    model.eval()
    for i in range(len(COCO_PATHS)):
        #evalute the dataset
        output_file = process_dataset_without_batches(model, device, COCO_PATHS[i], IMG_DIRS[i], DATASET_NAMES[i], 0.2)
        temp_filepath = os.path.abspath(f"tempfile_{test_name}_lr_{learning_rate}.json")
        output_file_no_coco_crap =[ x for x in output_file['annotations']]
        with open(temp_filepath, 'w') as f:
            json.dump(output_file_no_coco_crap, f)

        cocoEval = get_coco_metrics(COCO_PATHS[i], temp_filepath)
        ap_05 = cocoEval.stats[1]


        #check training_logging/{TEST_GROUP} dir exists. if not make it
        if not os.path.exists(f"training_logging/{test_group}/{test_name}"):
            os.makedirs(f"training_logging/{test_group}/{test_name}")

        if DATASET_NAMES[i] == MAV_VID_DATASET_NAME:
            mavvid_ap_val = ap_05
            # mavvid_ap_vals.append(ap_05)
            np.save(f"training_logging/{test_group}/{test_name}/{MAV_VID_DATASET_NAME}_{test_name}_lr_{learning_rate}_epoch_{epoch}.npy", cocoEval.stats)
        elif DATASET_NAMES[i] == DRONEBIRD_DATASET_NAME:
            dvb_ap_val = ap_05
            # dvb_ap_vals.append(ap_05)
            np.save(f"training_logging/{test_group}/{test_name}/{DRONEBIRD_DATASET_NAME}_{test_name}_lr_{learning_rate}_epoch_{epoch}.npy", cocoEval.stats)
        elif DATASET_NAMES[i] == ANTI_UAV_DATASET_NAME:
            antiuav_ap_val = ap_05
            # antiuav_ap_vals.append(ap_05)
            np.save(f"training_logging/{test_group}/{test_name}/{ANTI_UAV_DATASET_NAME}_{test_name}_lr_{learning_rate}_epoch_{epoch}.npy", cocoEval.stats)

        os.remove(temp_filepath)

    return mavvid_ap_val, dvb_ap_val, antiuav_ap_val