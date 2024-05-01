import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import random
from skimage.util import random_noise

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


class DroneDataset_Base(torch.utils.data.Dataset):
    def __init__(self, base_dir, local_transforms):
        self.base_dir = base_dir
        self.local_transforms = local_transforms
        self.DEBUG = False
        self.DEBUG_SAVE_IMAGES = False
        
        print("initializing dataset...")

        self.imgs = list(sorted(os.listdir(os.path.join(base_dir, "Images"))))
        self.masks = list(sorted(os.listdir(os.path.join(base_dir, "Masks"))))

        # self.imgs = list(sorted(os.listdir(os.path.join(base_dir, "Images"))))[:10000]
        # self.imgs = list(sorted(absoluteFilePaths(os.path.join(base_dir, "Images"))))[:10000] + list(sorted(absoluteFilePaths(os.path.join(other_dir, "Images"))))
        # print(self.imgs)
        # self.masks = list(sorted(absoluteFilePaths(os.path.join(base_dir, "Masks"))))[:10000]+ list(sorted(absoluteFilePaths(os.path.join(other_dir, "Masks"))))

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
            # If NOISE_ENABLED flag is true add noise to 50% of images
            #This is fucky, should add it as an image augmentation instead.
            if NOISE_ENABLED:
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
        #Set true to vizualise individual masks - saved in base.dir/debug_mask - make sure this dir exists
        # SAVE_IMAGES = False
        
        #get unique IDs
        obj_ids = np.unique(mask)
        #Only read from 2 onwards - before that we have background
        obj_ids = obj_ids[2:]
        
        #print(f"{obj_ids=}")
        
        #bird masks are above 176 (they are actually white but 177,178 after greyscale conversion)
        # bird_mask = mask >= 176
        #drone_mask = mask == drone_ids[:, None, None]
        
        #Drone masks are below 176 but above 1
        drone_mask = (mask > 2) 

        labeled_drone_mask = label(drone_mask)
        # print('Found ', len(np.unique(labeled_drone_mask)), ' connected drone masks')
        
        drone_masks_individual = []
        drone_boxes = []
        for drone in np.unique(labeled_drone_mask):
            if drone == 0:
                continue
            mask = (labeled_drone_mask==drone).astype(np.bool)
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



#This is a class for drones where they appear as random colours, NOT white
#This assumes that birds are white but does not output bird masks
class DroneDataset_DroneOnly(DroneDataset_Base):
    def __init__(self, base_dir, local_transforms):
        super().__init__(base_dir, local_transforms)


    # def __process_mask(self, mask):
    #     return super().__process_mask(mask)
    #     #Takes in the mask and processes it to output drone masks and bird masks
    def __process_mask(self, mask):
        #Set true to vizualise individual masks - saved in base.dir/debug_mask - make sure this dir exists
        SAVE_IMAGES = False
        
        #get unique IDs
        obj_ids = np.unique(mask)
        #Only read from 2 onwards - before that we have background
        obj_ids = obj_ids[2:]
        
        #print(f"{obj_ids=}")
        
        #bird masks are above 176 (they are actually white but 177,178 after greyscale conversion)
        # bird_mask = mask >= 176
        #drone_mask = mask == drone_ids[:, None, None]
        
        #Drone masks are below 176 but above 1
        drone_mask = ((mask > 1) * (mask < 176))
        #masks = mask == obj_ids[:, None, None]
        #(masks.size)

        #labeled_bird_mask = label(bird_mask)
        #print('Found ', len(np.unique(labeled_bird_mask)), ' connected bird masks')

        labeled_drone_mask = label(drone_mask)
        #print('Found ', len(np.unique(labeled_drone_mask)), ' connected drone masks')
        
        
        drone_masks_individual = []
        drone_boxes = []
        for drone in np.unique(labeled_drone_mask):
            if drone == 0:
                continue
            mask = (labeled_drone_mask==drone).astype(np.bool)
            if mask.sum() < MIN_DRONE_AREA:
                continue
                
            bbox = self.__generate_bbox(mask)
            
            #if bbox is none (i.e. it has no width/height, don't output bbox/mask)
            if bbox is None:
                continue

            #Add boxes and masks to the lists
            drone_boxes.append(bbox)
            drone_masks_individual.append(mask)

            if SAVE_IMAGES:
                mask = Image.fromarray(mask)
                mask.save(os.path.join(self.base_dir, "debug_mask", "_drone" + "_" + str(drone) + ".png"))


        return drone_masks_individual, drone_boxes


# #this dataset assumes white birds, and colourful drones on a black backgrond
# #
# class DroneDataset_DroneOnly(torch.utils.data.Dataset):
#     def __init__(self, base_dir, local_transforms):
#         self.base_dir = base_dir
#         self.local_transforms = local_transforms
#         self.DEBUG = True
        
#         print("initializing dataset...")
#         # other_dir = "/home/observation/repos/datasets/synth_masked_drones"

#         self.imgs = list(sorted(os.listdir(os.path.join(base_dir, "Images"))))[:5000]
#         self.masks = list(sorted(os.listdir(os.path.join(base_dir, "Masks"))))[:5000]

#         # self.imgs = list(sorted(os.listdir(os.path.join(base_dir, "Images"))))[:10000]
#         # self.imgs = list(sorted(absoluteFilePaths(os.path.join(base_dir, "Images"))))[:10000] + list(sorted(absoluteFilePaths(os.path.join(other_dir, "Images"))))
#         # print(self.imgs)
#         # self.masks = list(sorted(absoluteFilePaths(os.path.join(base_dir, "Masks"))))[:10000]+ list(sorted(absoluteFilePaths(os.path.join(other_dir, "Masks"))))

#         print("finished initializing")

        
#     def __getitem__(self,idx, convert_to_BW=False):
#         SAVE_IMAGES = False
#         DEBUG = False
        
#         # print("getting img id: {}".format(idx))
            
#         img_path = os.path.join(self.base_dir, "Images", self.imgs[idx])
#         mask_path = os.path.join(self.base_dir, "Masks", self.masks[idx])

#         # img_path = self.imgs[idx]
#         # mask_path = self.masks[idx]

#         #Load image
#         img = Image.open(img_path).convert("RGB")
#         debug_img = Image.open(img_path).convert("RGB")
#         #Load mask (in greyscale)
#         mask = Image.open(mask_path).convert("L")
#         w, h = img.size

#         #Save images for debug
#         if SAVE_IMAGES:
#             img.save(os.path.join(self.base_dir, "debug_mask", str(idx) + "_" + "original" + ".png"))
#             mask.save(os.path.join(self.base_dir, "debug_mask", str(idx) + "_" + "mask" + ".png"))

#         #convert mask from PIL to numpy
#         mask = np.array(mask)
#         if DEBUG:
#             print(f"{mask.shape=}")
#             print(f"{mask=}")
        
        
#         #Split the mask into indivual mask based on the blobs.
#         #We split it into two categories: birds and drones
#         drone_masks_list, drone_boxes_list = self.__process_mask(mask)
        
#         if DEBUG:
#             print(f"{len(drone_masks_list)=}")
#             print(f"{len(bird_masks_list)=}")
#             print(f"{len(drone_boxes_list)=}")
#             print(f"{len(bird_boxes_list)=}")
            
        
#         #at the moment the masks are lists - we want to concatenate them together 
#         masks = []
#         masks = drone_masks_list
            
#         boxes = []
#         boxes = drone_boxes_list

#         if DEBUG:
#             print("finished concatenating masks and boxes")
#             print("moving things to torch")

#         #convert masks to tensor to use masks_to_boxes function
#         masks = torch.as_tensor(np.array(masks), dtype=torch.bool)
#         # convert everything else into a torch.Tensor
#         # boxes = torch.as_tensor(boxes, dtype=torch.float32)



#         # there are two classes: bird masks and drone masks. create them then concat together
#         drone_labels = torch.ones((len(drone_masks_list),), dtype=torch.int64)
#         # bird_labels = torch.full((len(bird_masks_list),),2, dtype=torch.int64)
#         # labels = torch.cat([drone_labels, bird_labels])
#         labels = drone_labels
#         # labels = torch.sub(labels, 1)

#         # guard against no boxes via resizing
#         boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
#         # boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # boxes[:, 2:] += boxes[:, :2] - 01/06/23 - this is for the case where the boxes are in xywh format, but we have xyxy format.
#         #besides, i don't see the point of adding the width and height to the x and y coordinates, you are doubling the size of the box
#         #maybe there's a reason for this in the original coco ds but i don't know it
#         boxes[:, 0::2].clamp_(min=0, max=w)
#         boxes[:, 1::2].clamp_(min=0, max=h)
#         keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
#         boxes = boxes[keep]
#         labels = labels[keep]
#         masks = masks[keep]
        
#         if DEBUG:
#             print(f"{drone_labels=}")
#             print(f"{bird_labels=}")
#             print(f"{labels=}")
        
#         image_id = torch.tensor([idx])

#         #do a check if boxes is empty: if its empty make area = boxes so that it doesnt throw an error
#         if len(boxes) == 0:
#             print(f"image {idx} has no bounding boxes.")
#             area = boxes
#         else:
#             area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        
#         num_objs = len(drone_masks_list)
#         # suppose all instances are not crowd
#         iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
#         iscrowd = iscrowd[keep]

#         # print(f"{img.size=}")
#         # print(f"{masks.shape=}")
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["masks"] = masks
#         target["image_id"] = image_id
#         target["area"] = area
#         target["iscrowd"] = iscrowd

    
#         target['size'] = torch.as_tensor([int(h), int(w)])
#         target['orig_size'] = torch.as_tensor([int(h), int(w)])

#         if self.local_transforms is not None:
#             #move image from PIL to numpy
#             # for 50% of images, add noise 
#             if NOISE_ENABLED:
#                 if random.random() < 0.5:
#                     im_arr = np.asarray(img)
#                     randnum = random.randint(1,3)
#                     if randnum == 1:
#                         # noisetype = 's&p'
#                         strength = random.uniform(0,0.4)
#                         im_arr = random_noise(im_arr, mode='s&p', amount=strength)
#                     elif randnum == 2:
#                         # noisetype = 'gaussian'
#                         strength = random.uniform(0,1.0)
#                         im_arr = random_noise(im_arr, mode='gaussian', var=strength**2, clip=True)
#                     else:  
#                         # noisetype = 'poisson'
#                         im_arr = random_noise(im_arr, mode='poisson', clip=True)
#                     im_arr = (255*im_arr).astype(np.uint8)
#                     #now mvoe back to PIL
#                     img = Image.fromarray(im_arr)

#             img, target = self.local_transforms(img, target)

#         # new_bboxes = target["boxes"]
#         # keep2 = (new_bboxes[:, 3] > new_bboxes[:, 1]) & (new_bboxes[:, 2] > new_bboxes[:, 0])
#         # out_target["boxes"] = boxes[keep2]
#         # out_target["labels"] = labels[keep2]
#         # out_target["masks"] = masks[keep2]
#         # out_target["image_id"] = image_id[keep2]
#         # out_target["area"] = area[keep2]
#         # out_target["iscrowd"] = iscrowd[keep2]
#         # target['size'] = torch.as_tensor([int(h), int(w)])
#         # target['orig_size'] = torch.as_tensor([int(h), int(w)])

#         # boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # boxes[:, 2:] += boxes[:, :2]
#         # boxes[:, 0::2].clamp_(min=0, max=w)
#         # boxes[:, 1::2].clamp_(min=0, max=h)
#         # keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
#         # boxes = boxes[keep]
#         # labels = labels[keep]
#         # masks = masks[keep]


#         #print("Finishing getting img id: {}".format(idx))
#         return img, target


    
#     def __generate_bbox(self, mask):           
#         pos = np.where(mask)
#         xmin = np.min(pos[1])
#         xmax = np.max(pos[1])
#         ymin = np.min(pos[0])
#         ymax = np.max(pos[0])

#         #if bbox has no width/height, return None
#         if (xmin == xmax):
#             return None
#         if (ymin == ymax):
#             return None
        
#         return [xmin, ymin, xmax, ymax]
        

        
#     #Takes in the mask and processes it to output drone masks and bird masks
#     def __process_mask(self, mask):
#         #Set true to vizualise individual masks - saved in base.dir/debug_mask - make sure this dir exists
#         SAVE_IMAGES = False
        
#         #get unique IDs
#         obj_ids = np.unique(mask)
#         #Only read from 2 onwards - before that we have background
#         obj_ids = obj_ids[2:]
        
#         #print(f"{obj_ids=}")
        
#         #bird masks are above 176 (they are actually white but 177,178 after greyscale conversion)
#         bird_mask = mask >= 176
#         #drone_mask = mask == drone_ids[:, None, None]
        
#         #Drone masks are below 176 but above 1
#         drone_mask = ((mask > 1) * (mask < 176))
#         #masks = mask == obj_ids[:, None, None]
#         #(masks.size)

#         #labeled_bird_mask = label(bird_mask)
#         #print('Found ', len(np.unique(labeled_bird_mask)), ' connected bird masks')

#         labeled_drone_mask = label(drone_mask)
#         #print('Found ', len(np.unique(labeled_drone_mask)), ' connected drone masks')
        
        
#         # bird_masks_individual = []
#         # bird_boxes = []
#         # for bird in np.unique(labeled_bird_mask):
#         #     if bird == 0: # id = 0 is for background
#         #         continue

#         #     #print(f"{np.unique(labeled_bird_mask)=}{bird=}")
#         #     mask = (labeled_bird_mask==bird).astype(np.bool)
#         #     #if mask is smaller than 8 pixels, dont add it
#         #     if mask.sum() < 8:
#         #         continue

#         #     bbox = self.__generate_bbox(mask)
            
#         #     #if bbox is none (i.e. it has no width/height, don't output bbox/mask)
#         #     if bbox is None:
#         #         continue

#         #     #Add boxes and masks to the lists
#         #     bird_boxes.append(bbox)
#         #     bird_masks_individual.append(mask)
            
#         #     if SAVE_IMAGES:
#         #         mask = Image.fromarray(mask)
#         #         mask.save(os.path.join(self.base_dir, "debug_mask", "_bird" + "_" + str(bird) + ".png"))
            
        
#         drone_masks_individual = []
#         drone_boxes = []
#         for drone in np.unique(labeled_drone_mask):
#             if drone == 0:
#                 continue
#             mask = (labeled_drone_mask==drone).astype(np.bool)
#             if mask.sum() < MIN_DRONE_AREA:
#                 continue
                
#             bbox = self.__generate_bbox(mask)
            
#             #if bbox is none (i.e. it has no width/height, don't output bbox/mask)
#             if bbox is None:
#                 continue

#             #Add boxes and masks to the lists
#             drone_boxes.append(bbox)
#             drone_masks_individual.append(mask)

#             if SAVE_IMAGES:
#                 mask = Image.fromarray(mask)
#                 mask.save(os.path.join(self.base_dir, "debug_mask", "_drone" + "_" + str(drone) + ".png"))


#         return drone_masks_individual, drone_boxes
        
#     #The default way to process masks taken from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
#     #It doesnt actually work, because the Blender generated masks seem to span across multiple colours and break this method
#     #Use __process_mask instead
#     def __process_mask_default(self, mask):
#         obj_ids = np.unique(mask)
#         # first id is the background, so remove it
#         obj_ids = obj_ids[1:]
#         masks = mask == obj_ids[:, None, None]
        
#         colour_id = 0
#         for m in masks:
#            debug_mask = Image.fromarray(m)
#            debug_mask.save(os.path.join(self.base_dir, "debug_mask", str(idx) + "_" + str(colour_id) + ".png"))
#            debug_mask.show()
#            colour_id += 1

#     def get_boxes(self):
#         print("Getting boxes")
#         id = 0
#         #need to change dataset size here...
#         for id in range(0, 18006):
#             img, target = self.__getitem__(id)
#             # print("id: {}".format(id))
#             # print(target["boxes"])
#             # print(target["masks"])
#             # print(target["labels"])

#     def get_length(self):
#         return len(self.imgs)



#     def __len__(self):
#         return len(self.imgs)
    


# class DroneDataset(torch.utils.data.Dataset):
#     def __init__(self, base_dir, local_transforms):
#         self.base_dir = base_dir
#         self.local_transforms = local_transforms
#         self.DEBUG = True
        
#         print("initializing dataset...")
        

#         self.imgs = list(sorted(os.listdir(os.path.join(base_dir, "Images"))))[:10000]
#         self.masks = list(sorted(os.listdir(os.path.join(base_dir, "Masks"))))[:10000]

#         print("finished initializing")
#     def __generate_bbox(self, mask):           
#         pos = np.where(mask)
#         xmin = np.min(pos[1])
#         xmax = np.max(pos[1])
#         ymin = np.min(pos[0])
#         ymax = np.max(pos[0])

#         #if bbox has no width/height, return None
#         if (xmin == xmax):
#             return None
#         if (ymin == ymax):
#             return None
        
#         return [xmin, ymin, xmax, ymax]
        

        
#     #Takes in the mask and processes it to output drone masks and bird masks
#     def __process_mask(self, mask):
#         #Set true to vizualise individual masks - saved in base.dir/debug_mask - make sure this dir exists
#         SAVE_IMAGES = False
        
#         #get unique IDs
#         obj_ids = np.unique(mask)
#         #Only read from 2 onwards - before that we have background
#         obj_ids = obj_ids[2:]
        
#         #print(f"{obj_ids=}")
        
#         #bird masks are above 176 (they are actually white but 177,178 after greyscale conversion)
#         bird_mask = mask >= 176
#         #drone_mask = mask == drone_ids[:, None, None]
        
#         #Drone masks are below 176 but above 1
#         drone_mask = ((mask > 1) * (mask < 176))
#         #masks = mask == obj_ids[:, None, None]
#         #(masks.size)

#         labeled_bird_mask = label(bird_mask)
#         #print('Found ', len(np.unique(labeled_bird_mask)), ' connected bird masks')

#         labeled_drone_mask = label(drone_mask)
#         #print('Found ', len(np.unique(labeled_drone_mask)), ' connected drone masks')
        
        
#         bird_masks_individual = []
#         bird_boxes = []
#         for bird in np.unique(labeled_bird_mask):
#             if bird == 0: # id = 0 is for background
#                 continue

#             #print(f"{np.unique(labeled_bird_mask)=}{bird=}")
#             mask = (labeled_bird_mask==bird).astype(np.bool)
#             #if mask is smaller than 8 pixels, dont add it
#             if mask.sum() < MIN_DRONE_AREA:
#                 continue

#             bbox = self.__generate_bbox(mask)
            
#             #if bbox is none (i.e. it has no width/height, don't output bbox/mask)
#             if bbox is None:
#                 continue

#             #Add boxes and masks to the lists
#             bird_boxes.append(bbox)
#             bird_masks_individual.append(mask)
            
#             if SAVE_IMAGES:
#                 mask = Image.fromarray(mask)
#                 mask.save(os.path.join(self.base_dir, "debug_mask", "_bird" + "_" + str(bird) + ".png"))
            
        
#         drone_masks_individual = []
#         drone_boxes = []
#         for drone in np.unique(labeled_drone_mask):
#             if drone == 0:
#                 continue
#             mask = (labeled_drone_mask==drone).astype(np.bool)
#             if mask.sum() < MIN_DRONE_AREA:
#                 continue
                
#             bbox = self.__generate_bbox(mask)
            
#             #if bbox is none (i.e. it has no width/height, don't output bbox/mask)
#             if bbox is None:
#                 continue

#             #Add boxes and masks to the lists
#             drone_boxes.append(bbox)
#             drone_masks_individual.append(mask)

#             if SAVE_IMAGES:
#                 mask = Image.fromarray(mask)
#                 mask.save(os.path.join(self.base_dir, "debug_mask", "_drone" + "_" + str(drone) + ".png"))


#         return drone_masks_individual, bird_masks_individual, drone_boxes, bird_boxes
        
#     #The default way to process masks taken from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
#     #It doesnt actually work, because the Blender generated masks seem to span across multiple colours and break this method
#     #Use __process_mask instead
#     def __process_mask_default(self, mask):
#         obj_ids = np.unique(mask)
#         # first id is the background, so remove it
#         obj_ids = obj_ids[1:]
#         masks = mask == obj_ids[:, None, None]
        
#         colour_id = 0
#         for m in masks:
#            debug_mask = Image.fromarray(m)
#            debug_mask.save(os.path.join(self.base_dir, "debug_mask", str(idx) + "_" + str(colour_id) + ".png"))
#            debug_mask.show()
#            colour_id += 1
        
#     def __getitem__(self,idx, convert_to_BW=False):
#         SAVE_IMAGES = False
#         DEBUG = False
        
#         # print("getting img id: {}".format(idx))
            
#         img_path = os.path.join(self.base_dir, "Images", self.imgs[idx])
#         mask_path = os.path.join(self.base_dir, "Masks", self.masks[idx])

#         #Load image
#         img = Image.open(img_path).convert("RGB")
#         debug_img = Image.open(img_path).convert("RGB")
#         #Load mask (in greyscale)
#         mask = Image.open(mask_path).convert("L")

#         #Save images for debug
#         if SAVE_IMAGES:
#             img.save(os.path.join(self.base_dir, "debug_mask", str(idx) + "_" + "original" + ".png"))
#             mask.save(os.path.join(self.base_dir, "debug_mask", str(idx) + "_" + "mask" + ".png"))

#         #convert mask from PIL to numpy
#         mask = np.array(mask)
#         if DEBUG:
#             print(f"{mask.shape=}")
#             print(f"{mask=}")
        
        
#         #Split the mask into indivual mask based on the blobs.
#         #We split it into two categories: birds and drones
#         drone_masks_list, bird_masks_list, drone_boxes_list, bird_boxes_list = self.__process_mask(mask)
        
#         if DEBUG:
#             print(f"{len(drone_masks_list)=}")
#             print(f"{len(bird_masks_list)=}")
#             print(f"{len(drone_boxes_list)=}")
#             print(f"{len(bird_boxes_list)=}")
            
        
#         #at the moment the masks are lists - we want to concatenate them together 
#         masks = []
#         if not drone_masks_list:
#             masks = bird_masks_list
#         elif not bird_masks_list:
#             masks = drone_masks_list
#         else:
#             masks = np.concatenate([drone_masks_list, bird_masks_list], axis=0)
            
#         boxes = []
#         if not drone_boxes_list:
#             boxes = bird_boxes_list
#         elif not bird_boxes_list:
#             boxes = drone_boxes_list
#         else:
#             boxes = np.concatenate([drone_boxes_list, bird_boxes_list], axis=0)

        
#         if DEBUG:
#             print("finished concatenating masks and boxes")
#             print("moving things to torch")
        
#         # boxes = []
#         # for i in range(num_objs):
#         #     pos = np.where(masks[i])
#         #     xmin = np.min(pos[1])
#         #     xmax = np.max(pos[1])
#         #     ymin = np.min(pos[0])
#         #     ymax = np.max(pos[0])
#         #     boxes.append([xmin, ymin, xmax, ymax])

#         #     if (xmin == xmax):
#         #         print(f"img {idx} has a bounding box with the same x_min {xmin} and x_max {xmax}")
#         #     if (ymin == ymax):
#         #         print(f"img {idx} has a bounding box with the same y_min {ymin} and y_max {ymax}")

#         #convert masks to tensor to use masks_to_boxes function
#         masks = torch.as_tensor(np.array(masks), dtype=torch.bool)
#         # convert everything else into a torch.Tensor
#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # there are two classes: bird masks and drone masks. create them then concat together
#         drone_labels = torch.ones((len(drone_masks_list),), dtype=torch.int64)
#         bird_labels = torch.full((len(bird_masks_list),),2, dtype=torch.int64)
#         labels = torch.cat([drone_labels, bird_labels])
#         labels = torch.sub(labels, 1)
        
#         if DEBUG:
#             print(f"{drone_labels=}")
#             print(f"{bird_labels=}")
#             print(f"{labels=}")
        
#         image_id = torch.tensor([idx])

#         #do a check if boxes is empty: if its empty make area = boxes so that it doesnt throw an error
#         if len(boxes) == 0:
#             print(f"image {idx} has no bounding boxes.")
#             area = boxes
#         else:
#             area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        
#         num_objs = len(drone_masks_list) + len(bird_masks_list)
#         # suppose all instances are not crowd
#         iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

#         # print(f"{img.size=}")
#         # print(f"{masks.shape=}")
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["masks"] = masks
#         target["image_id"] = image_id
#         target["area"] = area
#         target["iscrowd"] = iscrowd

#         if self.local_transforms is not None:
#             img, target = self.local_transforms(img, target)

#         #print("Finishing getting img id: {}".format(idx))
#         return img, target

#     def get_boxes(self):
#         print("Getting boxes")
#         id = 0
#         #need to change dataset size here...
#         for id in range(0, 18006):
#             img, target = self.__getitem__(id)
#             # print("id: {}".format(id))
#             # print(target["boxes"])
#             # print(target["masks"])
#             # print(target["labels"])

#     def get_length(self):
#         return len(self.imgs)



#     def __len__(self):
#         return len(self.imgs)
    
    


# class DroneSegmentationDataset(torch.utils.data.Dataset):
#     def __init__(self, baseDir, local_transforms):
#         self.baseDir = baseDir
#         self.imagePaths = list(sorted(os.listdir(os.path.join(baseDir, "Images"))))
#         self.maskPaths = list(sorted(os.listdir(os.path.join(baseDir, "Masks"))))
#         self.local_transforms = local_transforms
        
#     def __len__(self):
#         return len(self.imagePaths)-5
    
#     def __getitem__(self,idx):
#         imagePath = self.baseDir + "Images/" + self.imagePaths[idx]
#         imagePath_2 = self.baseDir + "Images/" + self.imagePaths[idx + 1]
#         imagePath_3 = self.baseDir + "Images/" + self.imagePaths[idx + 2]
#         imagePath_4 = self.baseDir + "Images/" + self.imagePaths[idx + 3]
#         imagePath_5 = self.baseDir + "Images/" + self.imagePaths[idx + 4]
        
# #         print(f"{imagePath=}")
#         image = cv2.imread(imagePath)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
# #         print(f"{imagePath_2=}")
#         image_2 = cv2.imread(imagePath_2)
#         image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
        
# #         print(f"{imagePath_3=}")
#         image_3 = cv2.imread(imagePath_3)
#         image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB)
        
# #         print(f"{imagePath_4=}")
#         image_4 = cv2.imread(imagePath_4)
#         image_4 = cv2.cvtColor(image_4, cv2.COLOR_BGR2RGB)
        
# #         print(f"{self.maskPaths[idx]=}")
#         mask = cv2.imread(self.baseDir + "Masks/" + self.maskPaths[idx], 0)
#         mask_2 = cv2.imread(self.baseDir + "Masks/" + self.maskPaths[idx + 1], 0)
#         mask_3 = cv2.imread(self.baseDir + "Masks/" + self.maskPaths[idx + 2])
#         #mask_4 = cv2.imread(self.baseDir + "Masks/" + self.maskPaths[idx + 3], 0)
#         #mask_5 = cv2.imread(self.baseDir + "Masks/" + self.maskPaths[idx + 4], 0)
        
#         if self.local_transforms is not None:
#             # trans_RGB=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             # trans_gray=transforms.Normalize(mean=[0.485], std=[0.229])

#             image = self.local_transforms(image)
#             image_2 = self.local_transforms(image_2)
#             image_3 = self.local_transforms(image_3)
#             image_4 = self.local_transforms(image_4)
#             # image = trans_RGB(image)
#             # image_2 = trans_RGB(image_2)
#             # image_3 = trans_RGB(image_3)
#             # image_4 = trans_RGB(image_4)
#             #image_5 = self.local_transforms(image_5)
#             mask = self.local_transforms(mask)
#             mask_2 = self.local_transforms(mask_2)
#             mask_3 = self.local_transforms(mask_3)
#             # mask = trans_gray(mask)
#             # mask_2 = trans_gray(mask_2)
#             # mask_3 = trans_gray(mask_3)
            

#         #get the mask for the drone, by thresholding the red channel
#         mask_drone = (mask_3[2] >= 0.4).float()
#         #get the mask for the bird, by thresholding the red and blue channels
#         #The birds should be blue, drones should be white
#         mask_bird = ((mask_3[2] < 0.4) * (mask_3[0] > 0.5)).float()
        
#         mask_3 = torch.stack((mask_drone,mask_bird))
    
#         return (torch.cat((image,image_2,image_3), axis=0), image_4, mask_3)

# #         print(image.shape)
# #         output = np.concatenate((image,image_2), axis=0)
# #         output = torch.cat((image,image_2), axis=0)
# #         print(output.shape)
            
    
#         # return (torch.cat((image,image_2,image_3), axis=0), image_4, mask_3)
# #         return (torch.cat((image,image_2,image_3), axis=0), image_4, torch.cat((mask,mask_2,mask_3), axis=0))
#         #return image, image_2
#         #return ([image, image_2, image_3], [mask, mask_2, mask_3])
        

# class DroneTemporalDataset(torch.utils.data.Dataset):
#     def __init__(self, base_dir, transforms):
#         self.base_dir = base_dir
#         self.transforms = transforms
        
#         print("initializing dataset...")

#         self.imgs = list(sorted(os.listdir(os.path.join(base_dir, "Images"))))
#         self.masks = list(sorted(os.listdir(os.path.join(base_dir, "Masks"))))

#         print("finished initializing")

    

#     def __getitem__(self, idx, convert_to_BW=True):
#         print("getting img id: {}".format(idx))
#         img_path = os.path.join(self.base_dir, "Images", self.imgs[idx])
#         mask_path = os.path.join(self.base_dir, "Masks", self.masks[idx])


#         colour = "RGB"
#         if (convert_to_BW):
#             colour = "L"

#         img = Image.open(img_path).convert(colour)
#         debug_img = Image.open(img_path).convert(colour)
#         mask = Image.open(mask_path)
#         mask = np.array(mask)




#         if (convert_to_BW): 
#             mask[mask < 128] = 0
#             mask[mask >= 128] = 255
#             np.unique(mask)

#         obj_ids = np.unique(mask[:,:,0])

#         #remove the first object as its the background
#         obj_ids = obj_ids[1:]

#         # print(obj_ids)
        
#         #delete the element that == 14 becuase thats noise
#         # obj_ids = np.delete(obj_ids, np.where(obj_ids == 14))

#         masks = []

#         #for every colour in the img, create a binary mask
#         for obj_id in obj_ids:
#             output = np.where(mask[:,:,0] == obj_id, True, False)
#             masks.append(output)

#             #display the masks...
#             if (DEBUG):
#                 plt.imshow(output)
#                 plt.show()

#         # labels = ["camera", "body", "propellers"]

            
#         # get bounding box coordinates for each mask
#         num_objs = len(obj_ids)
#         boxes = []
#         for i in range(num_objs):
#             pos = np.where(masks[i])
#             xmin = np.min(pos[1])
#             xmax = np.max(pos[1])
#             ymin = np.min(pos[0])
#             ymax = np.max(pos[0])
#             boxes.append([xmin, ymin, xmax, ymax])

#             if (xmin == xmax):
#                 print("img {} has a bounding box with the same x_min and x_max".format(idx))

#             if (DEBUG):
#                 draw = ImageDraw.Draw(debug_img)
#                 draw.rectangle([xmin, ymin, xmax, ymax], fill=None, outline="rgb(255,0,0)")
#                 debug_img.save(os.path.join(self.base_dir, "debug", str(idx) + "_" + str(i) + ".png"))

#         # if (idx == 5):
#         #     print(boxes)



        
#         # convert everything into a torch.Tensor
#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # there is only one class
#         labels = torch.ones((num_objs,), dtype=torch.int64)
#         masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
#         image_id = torch.tensor([idx])

#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         # suppose all instances are not crowd
#         iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["masks"] = masks
#         target["image_id"] = image_id
#         target["area"] = area
#         target["iscrowd"] = iscrowd

#         if self.transforms is not None:
#             img, target = self.transforms(img, target)

#         print("Finishing getting img id: {}".format(idx))
#         return img, target

#     def get_boxes(self):
#         print("Getting boxes")
#         id = 0
#         for id in range(0, 1800):
#             img, target = self.__getitem__(id)
#             # print("id: {}".format(id))
#             # print(target["boxes"])
#             # print(target["masks"])
#             # print(target["labels"])




#     def get_length(self):
#         return len(self.imgs)



#     def __len__(self):
#         return len(self.imgs)



# class ValidationDataset(torch.utils.data.Dataset):
#     def __init__(self, base_dir, local_transforms):
#         self.base_dir = base_dir
#         self.local_transforms = local_transforms
#         self.DEBUG = False
        
#         print("initializing dataset...")

#         self.imgs = list(sorted(os.listdir(os.path.join(base_dir, "Images"))))
#         self.masks = list(sorted(os.listdir(os.path.join(base_dir, "Masks"))))

#         # self.imgs = list(sorted(os.listdir(os.path.join(base_dir, "Images"))))[:10000]
#         # self.imgs = list(sorted(absoluteFilePaths(os.path.join(base_dir, "Images"))))[:10000] + list(sorted(absoluteFilePaths(os.path.join(other_dir, "Images"))))
#         # print(self.imgs)
#         # self.masks = list(sorted(absoluteFilePaths(os.path.join(base_dir, "Masks"))))[:10000]+ list(sorted(absoluteFilePaths(os.path.join(other_dir, "Masks"))))

#         print("finished initializing")

        
#     def __getitem__(self,idx, convert_to_BW=False):
#         SAVE_IMAGES = False
#         DEBUG = False
        
#         # print("getting img id: {}".format(idx))
            
#         img_path = os.path.join(self.base_dir, "Images", self.imgs[idx])
#         mask_path = os.path.join(self.base_dir, "Masks", self.masks[idx])

#         # img_path = self.imgs[idx]
#         # mask_path = self.masks[idx]

#         #Load image
#         img = Image.open(img_path).convert("RGB")
#         debug_img = Image.open(img_path).convert("RGB")
#         #Load mask (in greyscale)
#         mask = Image.open(mask_path).convert("L")
#         w, h = img.size

#         #Save images for debug
#         if SAVE_IMAGES:
#             img.save(os.path.join(self.base_dir, "debug_mask", str(idx) + "_" + "original" + ".png"))
#             mask.save(os.path.join(self.base_dir, "debug_mask", str(idx) + "_" + "mask" + ".png"))

#         #convert mask from PIL to numpy
#         mask = np.array(mask)
#         if DEBUG:
#             print(f"{mask.shape=}")
#             print(f"{mask=}")
        
        
#         #Split the mask into indivual mask based on the blobs.
#         #We split it into two categories: birds and drones
#         drone_masks_list, drone_boxes_list = self.__process_mask(mask)
        
#         if DEBUG:
#             print(f"{len(drone_masks_list)=}")
#             print(f"{len(bird_masks_list)=}")
#             print(f"{len(drone_boxes_list)=}")
#             print(f"{len(bird_boxes_list)=}")
            
        
#         #at the moment the masks are lists - we want to concatenate them together 
#         masks = []
#         masks = drone_masks_list
            
#         boxes = []
#         boxes = drone_boxes_list

#         if DEBUG:
#             print("finished concatenating masks and boxes")
#             print("moving things to torch")

#         #convert masks to tensor to use masks_to_boxes function
#         masks = torch.as_tensor(np.array(masks), dtype=torch.bool)
#         # convert everything else into a torch.Tensor
#         # boxes = torch.as_tensor(boxes, dtype=torch.float32)



#         # there are two classes: bird masks and drone masks. create them then concat together
#         drone_labels = torch.ones((len(drone_masks_list),), dtype=torch.int64)
#         # bird_labels = torch.full((len(bird_masks_list),),2, dtype=torch.int64)
#         # labels = torch.cat([drone_labels, bird_labels])
#         labels = drone_labels
#         # labels = torch.sub(labels, 1)

#         # guard against no boxes via resizing
#         boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
#         # boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # boxes[:, 2:] += boxes[:, :2] - 01/06/23 - this is for the case where the boxes are in xywh format, but we have xyxy format.
#         #besides, i don't see the point of adding the width and height to the x and y coordinates, you are doubling the size of the box
#         #maybe there's a reason for this in the original coco ds but i don't know it
#         boxes[:, 0::2].clamp_(min=0, max=w)
#         boxes[:, 1::2].clamp_(min=0, max=h)
#         keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
#         boxes = boxes[keep]
#         labels = labels[keep]
#         masks = masks[keep]
        
#         if DEBUG:
#             print(f"{drone_labels=}")
#             print(f"{bird_labels=}")
#             print(f"{labels=}")
        
#         image_id = torch.tensor([idx])

#         #do a check if boxes is empty: if its empty make area = boxes so that it doesnt throw an error
#         if len(boxes) == 0:
#             print(f"image {idx} has no bounding boxes.")
#             area = boxes
#         else:
#             area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        
#         num_objs = len(drone_masks_list)
#         # suppose all instances are not crowd
#         iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
#         iscrowd = iscrowd[keep]

#         # print(f"{img.size=}")
#         # print(f"{masks.shape=}")
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["masks"] = masks
#         target["image_id"] = image_id
#         target["area"] = area
#         target["iscrowd"] = iscrowd

    
#         target['size'] = torch.as_tensor([int(h), int(w)])
#         target['orig_size'] = torch.as_tensor([int(h), int(w)])

#         if self.local_transforms is not None:
#             #move image from PIL to numpy
#             # for 50% of images, add noise 
#             if NOISE_ENABLED:
#                 if random.random() < 0.5:
#                     im_arr = np.asarray(img)
#                     randnum = random.randint(1,3)
#                     if randnum == 1:
#                         # noisetype = 's&p'
#                         strength = random.uniform(0,0.4)
#                         im_arr = random_noise(im_arr, mode='s&p', amount=strength)
#                     elif randnum == 2:
#                         # noisetype = 'gaussian'
#                         strength = random.uniform(0,1.0)
#                         im_arr = random_noise(im_arr, mode='gaussian', var=strength**2, clip=True)
#                     else:  
#                         # noisetype = 'poisson'
#                         im_arr = random_noise(im_arr, mode='poisson', clip=True)
#                     im_arr = (255*im_arr).astype(np.uint8)
#                     #now mvoe back to PIL
#                     img = Image.fromarray(im_arr)
            
#             img, target = self.local_transforms(img, target)

#         # new_bboxes = target["boxes"]
#         # keep2 = (new_bboxes[:, 3] > new_bboxes[:, 1]) & (new_bboxes[:, 2] > new_bboxes[:, 0])
#         # out_target["boxes"] = boxes[keep2]
#         # out_target["labels"] = labels[keep2]
#         # out_target["masks"] = masks[keep2]
#         # out_target["image_id"] = image_id[keep2]
#         # out_target["area"] = area[keep2]
#         # out_target["iscrowd"] = iscrowd[keep2]
#         # target['size'] = torch.as_tensor([int(h), int(w)])
#         # target['orig_size'] = torch.as_tensor([int(h), int(w)])

#         # boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # boxes[:, 2:] += boxes[:, :2]
#         # boxes[:, 0::2].clamp_(min=0, max=w)
#         # boxes[:, 1::2].clamp_(min=0, max=h)
#         # keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
#         # boxes = boxes[keep]
#         # labels = labels[keep]
#         # masks = masks[keep]


#         #print("Finishing getting img id: {}".format(idx))
#         return img, target


    
#     def __generate_bbox(self, mask):           
#         pos = np.where(mask)
#         xmin = np.min(pos[1])
#         xmax = np.max(pos[1])
#         ymin = np.min(pos[0])
#         ymax = np.max(pos[0])

#         #if bbox has no width/height, return None
#         if (xmin == xmax):
#             return None
#         if (ymin == ymax):
#             return None
        
#         return [xmin, ymin, xmax, ymax]
        

        
#     #Takes in the mask and processes it to output drone masks and bird masks
#     def __process_mask(self, mask):
#         #Set true to vizualise individual masks - saved in base.dir/debug_mask - make sure this dir exists
#         SAVE_IMAGES = False
        
#         #get unique IDs
#         obj_ids = np.unique(mask)
#         #Only read from 2 onwards - before that we have background
#         obj_ids = obj_ids[2:]
        
#         #print(f"{obj_ids=}")
        
#         #bird masks are above 176 (they are actually white but 177,178 after greyscale conversion)
#         # bird_mask = mask >= 176
#         #drone_mask = mask == drone_ids[:, None, None]
        
#         #Drone masks are below 176 but above 1
#         drone_mask = (mask > 2) 
#         #masks = mask == obj_ids[:, None, None]
#         #(masks.size)

#         #labeled_bird_mask = label(bird_mask)
#         #print('Found ', len(np.unique(labeled_bird_mask)), ' connected bird masks')

#         labeled_drone_mask = label(drone_mask)
#         #print('Found ', len(np.unique(labeled_drone_mask)), ' connected drone masks')
        
        
#         # bird_masks_individual = []
#         # bird_boxes = []
#         # for bird in np.unique(labeled_bird_mask):
#         #     if bird == 0: # id = 0 is for background
#         #         continue

#         #     #print(f"{np.unique(labeled_bird_mask)=}{bird=}")
#         #     mask = (labeled_bird_mask==bird).astype(np.bool)
#         #     #if mask is smaller than 8 pixels, dont add it
#         #     if mask.sum() < 8:
#         #         continue

#         #     bbox = self.__generate_bbox(mask)
            
#         #     #if bbox is none (i.e. it has no width/height, don't output bbox/mask)
#         #     if bbox is None:
#         #         continue

#         #     #Add boxes and masks to the lists
#         #     bird_boxes.append(bbox)
#         #     bird_masks_individual.append(mask)
            
#         #     if SAVE_IMAGES:
#         #         mask = Image.fromarray(mask)
#         #         mask.save(os.path.join(self.base_dir, "debug_mask", "_bird" + "_" + str(bird) + ".png"))
            
        
#         drone_masks_individual = []
#         drone_boxes = []
#         for drone in np.unique(labeled_drone_mask):
#             if drone == 0:
#                 continue
#             mask = (labeled_drone_mask==drone).astype(np.bool)
#             if mask.sum() < MIN_DRONE_AREA:
#                 continue
                
#             bbox = self.__generate_bbox(mask)
            
#             #if bbox is none (i.e. it has no width/height, don't output bbox/mask)
#             if bbox is None:
#                 continue

#             #Add boxes and masks to the lists
#             drone_boxes.append(bbox)
#             drone_masks_individual.append(mask)

#             if SAVE_IMAGES:
#                 mask = Image.fromarray(mask)
#                 mask.save(os.path.join(self.base_dir, "debug_mask", "_drone" + "_" + str(drone) + ".png"))


#         return drone_masks_individual, drone_boxes
        
#     #The default way to process masks taken from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
#     #It doesnt actually work, because the Blender generated masks seem to span across multiple colours and break this method
#     #Use __process_mask instead
#     def __process_mask_default(self, mask):
#         obj_ids = np.unique(mask)
#         # first id is the background, so remove it
#         obj_ids = obj_ids[1:]
#         masks = mask == obj_ids[:, None, None]
        
#         colour_id = 0
#         for m in masks:
#            debug_mask = Image.fromarray(m)
#            debug_mask.save(os.path.join(self.base_dir, "debug_mask", str(idx) + "_" + str(colour_id) + ".png"))
#            debug_mask.show()
#            colour_id += 1

#     def get_boxes(self):
#         print("Getting boxes")
#         id = 0
#         #need to change dataset size here...
#         for id in range(0, 18006):
#             img, target = self.__getitem__(id)
#             # print("id: {}".format(id))
#             # print(target["boxes"])
#             # print(target["masks"])
#             # print(target["labels"])

#     def get_length(self):
#         return len(self.imgs)



#     def __len__(self):
#         return len(self.imgs)