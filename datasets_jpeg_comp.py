import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import random
from skimage.util import random_noise
import imgaug.augmenters as iaa
from imgaug.augmenters.arithmetic import JpegCompression

#------------------------------------DATASET
from torchvision import transforms
from torch.utils.data import Dataset
import cv2


#for drone dataset
from skimage.morphology import label
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision.ops import masks_to_boxes

from config import MIN_DRONE_AREA, NOISE_ENABLED


def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))



#Dataset class which contains the JPEG augmentation.
#This was a temporary solution but ended up being the main solution - should be merged back into the datasets.py
#Both the JPEG + Noise augmentations should be moved to transforms.py

class DroneDataset_Base_JPEG(torch.utils.data.Dataset):
    def __init__(self, base_dir, local_transforms, jpeg_compression, noise, dataset_size = None):
        self.base_dir = base_dir
        self.local_transforms = local_transforms
        self.jpeg_compression = jpeg_compression
        self.noise = noise

        self.DEBUG = False
        self.DEBUG_SAVE_IMAGES = False
        
        print("initializing dataset...")

        if dataset_size is not None:
            self.imgs = list(sorted(os.listdir(os.path.join(base_dir, "Images"))))[:dataset_size]
            self.masks = list(sorted(os.listdir(os.path.join(base_dir, "Masks"))))[:dataset_size]
        else:
            self.imgs = list(sorted(os.listdir(os.path.join(base_dir, "Images"))))
            self.masks = list(sorted(os.listdir(os.path.join(base_dir, "Masks"))))

        print("finished initializing")

        
    def __getitem__(self,idx, convert_to_BW=False):
        #local debug flags!
        #SAVE_IMAGES saves the loaded images to a debug_mask directory. Useful to make sure everything is going well when testing a new dataset
        #DEBUG enables printing of debug msgs. Disabled by default but useful for finding if there are dimension mismatches etc. 
        #(Should move to proper logging really, instead of prints)
        SAVE_IMAGES = False
        DEBUG = False
        
        if DEBUG:
            print("getting img id: {}".format(idx))
            
        img_path = os.path.join(self.base_dir, "Images", self.imgs[idx])
        mask_path = os.path.join(self.base_dir, "Masks", self.masks[idx])

        # img_path = self.imgs[idx]
        # mask_path = self.masks[idx]

        #Load image
        img = Image.open(img_path).convert("RGB")
        debug_img = Image.open(img_path).convert("RGB")
        #Load mask (in greyscale)
        mask = Image.open(mask_path).convert("L")
        w, h = img.size

        #Save images for debug
        if SAVE_IMAGES:
            img.save(os.path.join(self.base_dir, "debug_mask", str(idx) + "_" + "original" + ".png"))
            mask.save(os.path.join(self.base_dir, "debug_mask", str(idx) + "_" + "mask" + ".png"))

        #convert mask from PIL to numpy
        mask = np.array(mask)
        if DEBUG:
            print(f"{mask.shape=}")
            print(f"{mask=}")
        
        
        #Split the mask into indivual mask based on the blobs.
        #We split it into two categories: birds and drones
        drone_masks_list, drone_boxes_list = self.__process_mask(mask)
        
        if DEBUG:
            print(f"{len(drone_masks_list)=}")
            print(f"{len(bird_masks_list)=}")
            print(f"{len(drone_boxes_list)=}")
            print(f"{len(bird_boxes_list)=}")
            
        
        #at the moment the masks are lists - we want to concatenate them together 
        masks = []
        masks = drone_masks_list
            
        boxes = []
        boxes = drone_boxes_list

        if DEBUG:
            print("finished concatenating masks and boxes")
            print("moving things to torch")

        #convert masks to tensor to use masks_to_boxes function
        masks = torch.as_tensor(np.array(masks), dtype=torch.bool)
        # convert everything else into a torch.Tensor
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)



        # there are two classes: bird masks and drone masks. create them then concat together
        drone_labels = torch.ones((len(drone_masks_list),), dtype=torch.int64)
        # bird_labels = torch.full((len(bird_masks_list),),2, dtype=torch.int64)
        # labels = torch.cat([drone_labels, bird_labels])
        labels = drone_labels
        # labels = torch.sub(labels, 1)

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # boxes[:, 2:] += boxes[:, :2] - 01/06/23 - this is for the case where the boxes are in xywh format, but we have xyxy format.
        #besides, i don't see the point of adding the width and height to the x and y coordinates, you are doubling the size of the box
        #maybe there's a reason for this in the original coco ds but i don't know it
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        masks = masks[keep]
        
        if DEBUG:
            print(f"{drone_labels=}")
            print(f"{bird_labels=}")
            print(f"{labels=}")
        
        image_id = torch.tensor([idx])

        #do a check if boxes is empty: if its empty make area = boxes so that it doesnt throw an error
        if len(boxes) == 0:
            print(f"image {idx} has no bounding boxes.")
            area = boxes
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        
        num_objs = len(drone_masks_list)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        iscrowd = iscrowd[keep]

        # print(f"{img.size=}")
        # print(f"{masks.shape=}")
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

    
        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])

        if self.local_transforms is not None:
            #move image from PIL to numpy
            # for 50% of images, add noise 
            if self.noise:
                if random.random() < 0.5:
                    im_arr = np.asarray(img)
                    randnum = random.randint(1,3)
                    if randnum == 1:
                        # noisetype = 's&p'
                        strength = random.uniform(0,0.4)
                        im_arr = random_noise(im_arr, mode='s&p', amount=strength)
                    elif randnum == 2:
                        # noisetype = 'gaussian'
                        strength = random.uniform(0,1.0)
                        im_arr = random_noise(im_arr, mode='gaussian', var=strength**2, clip=True)
                    else:  
                        # noisetype = 'poisson'
                        im_arr = random_noise(im_arr, mode='poisson', clip=True)
                    im_arr = (255*im_arr).astype(np.uint8)
                    #now mvoe back to PIL
                    img = Image.fromarray(im_arr)

            if self.jpeg_compression:
                if random.random() < 0.5:
                    jpeg_comp = iaa.JpegCompression(compression=(0, 100))
                    img = iaa.JpegCompression.augment_image(jpeg_comp, np.asarray(img))

                    img = Image.fromarray(img)
                    
            
            img, target = self.local_transforms(img, target)

        #print("Finishing getting img id: {}".format(idx))
        return img, target
    
    def __generate_bbox(self, mask):
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        #if bbox has no width/height, return None
        if (xmin == xmax):
            return None
        if (ymin == ymax):
            return None
        
        return [xmin, ymin, xmax, ymax]
    
        
    #Takes in the mask and processes it to output drone masks and bird masks
    def __process_mask(self, mask):       
        #get unique IDs
        obj_ids = np.unique(mask)
        #Only read from 2 onwards - before that we have background
        obj_ids = obj_ids[2:]
        
        #Drone masks are white - so above 176
        drone_mask = (mask > 176) 

        labeled_drone_mask = label(drone_mask)
        
        drone_masks_individual = []
        drone_boxes = []
        for drone in np.unique(labeled_drone_mask):
            if drone == 0:
                continue
            mask = (labeled_drone_mask==drone).astype(bool)
            if mask.sum() < MIN_DRONE_AREA:
                continue
                
            bbox = self.__generate_bbox(mask)
            
            #if bbox is none (i.e. it has no width/height, don't output bbox/mask)
            if bbox is None:
                continue

            #Add boxes and masks to the lists
            drone_boxes.append(bbox)
            drone_masks_individual.append(mask)

            if self.DEBUG_SAVE_IMAGES:
                mask = Image.fromarray(mask)
                mask.save(os.path.join(self.base_dir, "debug_mask", "_drone" + "_" + str(drone) + ".png"))


        return drone_masks_individual, drone_boxes
        
    #The default way to process masks taken from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    #It doesnt actually work, because the Blender generated masks seem to span across multiple colours and break this method
    #Use __process_mask instead
    def __process_mask_default(self, mask):
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        
        colour_id = 0
        for m in masks:
           debug_mask = Image.fromarray(m)
           debug_mask.save(os.path.join(self.base_dir, "debug_mask", str(idx) + "_" + str(colour_id) + ".png"))
           debug_mask.show()
           colour_id += 1

    def get_boxes(self):
        print("Getting boxes")
        id = 0
        #need to change dataset size here...
        for id in range(0, 18006):
            img, target = self.__getitem__(id)
            # print("id: {}".format(id))
            # print(target["boxes"])
            # print(target["masks"])
            # print(target["labels"])

    def get_length(self):
        return len(self.imgs)



    def __len__(self):
        return len(self.imgs)


