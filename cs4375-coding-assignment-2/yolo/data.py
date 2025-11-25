"""
CS 6375 Homework 2 Programming
Implement the __getitem__() function in this python script
"""
import torch
import torch.utils.data as data
import csv
import os, math
import sys
import time
import random
import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# The dataset class
class CrackerBox(data.Dataset):
    def __init__(self, image_set = 'train', data_path = 'data'):

        self.name = 'cracker_box_' + image_set
        self.image_set = image_set
        self.data_path = data_path
        self.classes = ('__background__', 'cracker_box')
        self.width = 640
        self.height = 480
        self.yolo_image_size = 448
        self.scale_width = self.yolo_image_size / self.width
        self.scale_height = self.yolo_image_size / self.height
        self.yolo_grid_num = 7
        self.yolo_grid_size = self.yolo_image_size / self.yolo_grid_num
        # split images into training set and validation set
        self.gt_files_train, self.gt_files_val = self.list_dataset()
        # the pixel mean for normalization
        self.pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)

        # training set
        if image_set == 'train':
            self.size = len(self.gt_files_train)
            self.gt_paths = self.gt_files_train
            print('%d images for training' % self.size)
        else:
            # validation set
            self.size = len(self.gt_files_val)
            self.gt_paths = self.gt_files_val
            print('%d images for validation' % self.size)


    # list the ground truth annotation files
    # use the first 100 images for training
    def list_dataset(self):
    
        filename = os.path.join(self.data_path, '*.txt')
        gt_files = sorted(glob.glob(filename))
        
        gt_files_train = gt_files[:100]
        gt_files_val = gt_files[100:]
        
        return gt_files_train, gt_files_val


    # TODO: implement this function
    def __getitem__(self, idx):
    
        # gt file
        filename_gt = self.gt_paths[idx]
        
        ### ADD YOUR CODE HERE ###

        base_name = os.path.basename(filename_gt)
        base_name = base_name.replace('-box.txt', '')
        # try multiple possible filenames
        possible_images = [
            base_name + '-color.jpg',
            base_name + '-gt.jpg',
            base_name + '.jpg'
        ]   

        image = None
        for name in possible_images:
            path = os.path.join(self.data_path, name)
            if os.path.exists(path):
                image_path = path
                image = cv2.imread(path)
            break

        if image is None:
            raise FileNotFoundError("No matching image for: " + filename_gt)

        image_path = os.path.join(self.data_path, possible_images)

    # read image
        image = cv2.imread(image_path)
        if image is None:
            print('Error: cannot load image %s' % image_path)
            sys.exit(1)

    # convert to RGB/ float32
        image = image[:, :, (2, 1, 0)].astype(np.float32)

    # read (cx, cy, w, h) in the original resolution
        with open(filename_gt, 'r') as f:
            line = f.readline().strip().split()
            cx = float(line[0])
            cy = float(line[1])
            w  = float(line[2])
            h  = float(line[3])

        image_resized = cv2.resize(image, (self.yolo_image_size, self.yolo_image_size))

    #subtract pixel mean, then divide by 255
        image_normalized = (image_resized - self.pixel_mean) / 255.0

    # convert HWC to CHW 
        image_blob = image_normalized.transpose((2, 0, 1))
        image_blob = torch.from_numpy(image_blob).float()

        cx_scaled = cx * self.scale_width
        cy_scaled = cy * self.scale_height
        w_scaled  = w  * self.scale_width
        h_scaled  = h  * self.scale_height

    # find which grid cell (7x7) contains the center point
        grid_x = int(cx_scaled / self.yolo_grid_size)
        grid_y = int(cy_scaled / self.yolo_grid_size)

    # normalized coordinates inside the selected grid cell
        cx_cell = (cx_scaled - grid_x * self.yolo_grid_size) / self.yolo_grid_size
        cy_cell = (cy_scaled - grid_y * self.yolo_grid_size) / self.yolo_grid_size
        w_cell  = w_scaled / self.yolo_image_size
        h_cell  = h_scaled / self.yolo_image_size

        gt_box = np.zeros((4, self.yolo_grid_num, self.yolo_grid_num), dtype=np.float32)

        gt_mask = np.zeros((self.yolo_grid_num, self.yolo_grid_num), dtype=np.float32)

        gt_box[0, grid_y, grid_x] = cx_cell
        gt_box[1, grid_y, grid_x] = cy_cell
        gt_box[2, grid_y, grid_x] = w_cell
        gt_box[3, grid_y, grid_x] = h_cell

        gt_mask[grid_y, grid_x] = 1.0

    # convert numpy arrays to torch tensors
        gt_box_blob = torch.from_numpy(gt_box).float()
        gt_mask_blob = torch.from_numpy(gt_mask).float()
        
        

        # this is the sample dictionary to be returned from this function
        sample = {'image': image_blob,
                  'gt_box': gt_box_blob,
                  'gt_mask': gt_mask_blob}

        return sample


    # len of the dataset
    def __len__(self):
        return self.size
        

# draw grid on images for visualization
def draw_grid(image, line_space=64):
    H, W = image.shape[:2]
    image[0:H:line_space] = [255, 255, 0]
    image[:, 0:W:line_space] = [255, 255, 0]


# the main function for testing
if __name__ == '__main__':
    dataset_train = CrackerBox('train', data_path='yolo/data')
    dataset_val   = CrackerBox('val', data_path='yolo/data')
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)
    
    # visualize the training data
    for i, sample in enumerate(train_loader):
        
        image = sample['image'][0].numpy().transpose((1, 2, 0))
        gt_box = sample['gt_box'][0].numpy()
        gt_mask = sample['gt_mask'][0].numpy()

        y, x = np.where(gt_mask == 1)
        cx = gt_box[0, y, x] * dataset_train.yolo_grid_size + x * dataset_train.yolo_grid_size
        cy = gt_box[1, y, x] * dataset_train.yolo_grid_size + y * dataset_train.yolo_grid_size
        w = gt_box[2, y, x] * dataset_train.yolo_image_size
        h = gt_box[3, y, x] * dataset_train.yolo_image_size

        x1 = cx - w * 0.5
        x2 = cx + w * 0.5
        y1 = cy - h * 0.5
        y2 = cy + h * 0.5

        print(image.shape, gt_box.shape)
        
        # visualization
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        im = image * 255.0 + dataset_train.pixel_mean
        im = im.astype(np.uint8)
        plt.imshow(im[:, :, (2, 1, 0)])
        plt.title('input image (448x448)', fontsize = 16)

        ax = fig.add_subplot(1, 3, 2)
        draw_grid(im)
        plt.imshow(im[:, :, (2, 1, 0)])
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor="none")
        ax.add_patch(rect)
        plt.plot(cx, cy, 'ro', markersize=12)
        plt.title('Ground truth bounding box in YOLO format', fontsize=16)
        
        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(gt_mask)
        plt.title('Ground truth mask in YOLO format (7x7)', fontsize=16)
        plt.show()
