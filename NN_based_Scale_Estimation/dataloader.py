import os
import os.path
import numpy as np
import random
import math
import datetime

from PIL import Image   # Load images from the dataset

import torch.utils.data
import torchvision.transforms as tranforms

class voDataLoader(torch.utils.data.Dataset):

    def __init__(self, img_dataset_path, pose_dataset_path, 
                       transform=None,  
                       sequence=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']):

        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path

        self.sequence = sequence

        self.len = 0    # The size of dataset in use

        self.transform = transform      # Image transformation conditions (Resolution Change)

        self.sequence_change = False

        self.sequence_idx = 0
        self.data_idx = 1        # Since DeepVO requires 2 consecutive images, Data Index starts from 1
        self.sequence_num = len(self.sequence)
        self.current_sequence_data_num = len(sorted(os.listdir(self.img_dataset_path + '/' + self.sequence[self.sequence_idx] + '/image_2')))
        self.img_path = sorted(os.listdir(self.img_dataset_path + '/' + self.sequence[self.sequence_idx] + '/image_2'))

        # Read 0th pose data and save it as current pose value
        self.pose_file = open(self.pose_dataset_path + '/' + self.sequence[self.sequence_idx] + '.txt', 'r')
        line = self.pose_file.readline()
        pose = line.strip().split()
        self.current_pose_T = np.array([float(pose[3]), float(pose[7]), float(pose[11])])
        self.prev_pose_T = np.array([0.0, 0.0, 0.0])

        for i in range(len(self.sequence)):            
            
            f = open(self.pose_dataset_path + '/' + self.sequence[i] + '.txt', 'r')
            while True:
                line = f.readline()
                if not line: break

                self.len += 1
            self.len -= 1
            f.close()

        print('Sequence in Use : {}'.format(self.sequence))
        print('Size of dataset : {}'.format(self.len))

    # Dataset Load Function
    def load_data(self):
        ### Dataset Image Preparation ###
        # Load Image at t-1 and t
        base_path = self.img_dataset_path + '/' + self.sequence[self.sequence_idx] + '/image_2'

        prev_img = Image.open(base_path + '/' + self.img_path[self.data_idx-1]).convert('RGB')
        current_img = Image.open(base_path + '/' + self.img_path[self.data_idx]).convert('RGB')

        # Transform the image according to the transformation conditions
        if self.transform is not None:
            prev_img = self.transform(prev_img)
            current_img = self.transform(current_img)

        ### Pose Data (Pose difference/change between t-1 and t) Preparation ###
        # Save previous groundtruth as t-1 value
        self.prev_pose_T = self.current_pose_T

        # Load groundtruth at t
        line = self.pose_file.readline()
        pose = line.strip().split()
        
        self.current_pose_T = np.array([float(pose[3]), float(pose[7]), float(pose[11])])

        #########################################################################

        self.data_idx += 1   # Increase data index everytime data is consumed by DeepVO network

        # Stack the image as indicated in DeepVO paper
        prev_current_stacked_img = np.asarray(np.concatenate([prev_img, current_img], axis=0))

        # Prepare 6 DOF pose vector between t-1 and t (dX dY dZ dRoll dPitch dYaw)
        grountruth_absoluateScale = np.linalg.norm(self.current_pose_T - self.prev_pose_T)

        return prev_current_stacked_img, grountruth_absoluateScale

    def __getitem__(self, index):
        
        # index is dummy value for pytorch
        
        # 'sequence_idx' and 'data_idx' are actual indices of dataset for training and testing.
        # Since the dataset is composed of multiple separate dataset sequences, sequence and data need their own indexing.
        
        # 'sequence_idx' is the index of KITTI dataset sequence in training or testing -> Dataset in Use
        # 'data_idx' is the index of data in the KITTI dataset sequence in training or testing -> Actual Data
        
        # Index reset if the index of data goes over the range of current dataset sequence
        if self.data_idx >= self.current_sequence_data_num:
            
            self.sequence_idx += 1
            
            if self.sequence_idx < self.sequence_num:

                self.current_sequence_data_num = len(sorted(os.listdir(self.img_dataset_path + '/' + self.sequence[self.sequence_idx] + '/image_2')))
                self.data_idx = 1

                self.img_path = sorted(os.listdir(self.img_dataset_path + '/' + self.sequence[self.sequence_idx] + '/image_2'))

                # current pose data reset for new sequence
                self.pose_file.close()
                self.pose_file = open(self.pose_dataset_path + '/' + self.sequence[self.sequence_idx] + '.txt', 'r')
                line = self.pose_file.readline()
                pose = line.strip().split()

                self.current_pose_T = np.array([float(pose[3]), float(pose[7]), float(pose[11])])
                self.prev_pose_T = np.array([0.0, 0.0, 0.0])

                print('[Dataset Sequence Change] : {}'.format(self.sequence[self.sequence_idx]))
                
                self.sequence_change = True     # Notify the network that sequence has changed

                prev_current_stacked_img, prev_current_odom = self.load_data()

                return prev_current_stacked_img, prev_current_odom
        else:

            prev_current_stacked_img, prev_current_odom = self.load_data()

            self.sequence_change = False    # Notify the network that sequence has not changed

            return prev_current_stacked_img, prev_current_odom

    def __len__(self):

        return self.len

    def reset_loader(self):

        self.sequence_idx = 0
        self.data_idx = 1        # Since DeepVO requires 2 consecutive images, Data Index starts from 1
        self.sequence_num = len(self.sequence)
        self.current_sequence_data_num = len(sorted(os.listdir(self.img_dataset_path + '/' + self.sequence[self.sequence_idx] + '/image_2')))
        self.img_path = sorted(os.listdir(self.img_dataset_path + '/' + self.sequence[self.sequence_idx] + '/image_2'))

        # Read 0th pose data and save it as current pose value
        self.pose_file = open(self.pose_dataset_path + '/' + self.sequence[self.sequence_idx] + '.txt', 'r')
        line = self.pose_file.readline()
        pose = line.strip().split()

        self.current_pose_T = np.array([float(pose[3]), float(pose[7]), float(pose[11])])
        self.prev_pose_T = np.array([0.0, 0.0, 0.0])
