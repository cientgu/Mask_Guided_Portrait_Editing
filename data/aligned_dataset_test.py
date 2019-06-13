### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

#aligned_dataset_test2image_bg.py

import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import time
import torch

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        # if opt.isTrain or self.opt.random_embed==False:
        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
        self.B_paths = sorted(make_dataset(self.dir_B))


        ### input A (label maps)
        dir_mask_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_mask_A = os.path.join(opt.dataroot, opt.phase2 + dir_mask_A)
        self.mask_A_paths = sorted(make_dataset(self.dir_mask_A))

        ### input B (real images)
        # if opt.isTrain or self.opt.random_embed==False:
        dir_mask_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_mask_B = os.path.join(opt.dataroot, opt.phase2 + dir_mask_B)  
        self.mask_B_paths = sorted(make_dataset(self.dir_mask_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]
        A = Image.open(A_path)        
        w,h = A.size
        max_size = max(w,h)

        if self.opt.longSize != max_size:
            scale_size = float(self.opt.longSize/max_size)
            new_w = int(scale_size * w)
            new_h = int(scale_size * h)
            A = A.resize((new_w,new_h),Image.NEAREST)
            # if self.opt.isTrain or self.opt.random_embed==False:
            B_path = self.B_paths[index]
            B = Image.open(B_path).convert('RGB')
            B = B.resize((new_w,new_h),Image.BICUBIC)
        else:
            # if self.opt.isTrain or self.opt.random_embed==False:
            B_path = self.B_paths[index]
            B = Image.open(B_path).convert('RGB')

        C_tensor = 0

        A_tensor = transforms.functional.to_tensor(A) * 255.0
        B_tensor = transforms.functional.to_tensor(B)
        real_B_tensor = B_tensor.clone()
        mask_bg = (A_tensor==0).type(torch.FloatTensor)
        B_tensor = torch.clamp(B_tensor + mask_bg*torch.ones(A_tensor.size()),0,1)
        B = transforms.functional.to_pil_image(B_tensor)                                      


        if self.opt.data_augmentation == True:
            assert self.opt.isTrain == True
            rotate,scale,shear = random.random()-0.5, random.random()-0.5, random.random()-0.5
            rotate,scale,shear = 0,0,0
            B = transforms.functional.affine(B, 20*rotate,[0,0],1+0.2*scale,10*shear,resample=Image.BICUBIC)
            A = transforms.functional.affine(A, 20*rotate,[0,0],1+0.2*scale,10*shear,resample=Image.NEAREST)
            C_tensor = transforms.functional.to_tensor(B)
            C_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C_tensor)
        

        # if self.opt.isTrain or self.opt.random_embed==False:
        B_tensor = transforms.functional.to_tensor(B)
        B_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B_tensor)
        real_B_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(real_B_tensor)

        # else:
            # B_tensor = 0
        
        # get mean of left eye, right eye, mouth 
        # first y next x

        A_tensor = transforms.functional.to_tensor(A) * 255.0

        mask_tensor = torch.zeros(6)
        try:
            mask_left_eye_r = torch.nonzero(A_tensor==4)
            this_top = int(torch.min(mask_left_eye_r,0)[0][1])
            this_left = int(torch.min(mask_left_eye_r,0)[0][2])
            this_bottom = int(torch.max(mask_left_eye_r,0)[0][1])
            this_right = int(torch.max(mask_left_eye_r,0)[0][2])
            x_mean = int((this_left+this_right)/2)
            y_mean = int((this_top+this_bottom)/2)
            mask_tensor[0] = y_mean
            mask_tensor[1] = x_mean
            # mask_list.append(x_mean)
            # mask_list.append(y_mean)
        except:
            print("left eye problem ------------------")
            print(A_path)
            mask_tensor[0] = 116
            mask_tensor[1] = 96
            # mask_list.append(116)
            # mask_list.append(96)

        try:
            mask_right_eye_r = torch.nonzero(A_tensor==5)
            this_top = int(torch.min(mask_right_eye_r,0)[0][1])
            this_left = int(torch.min(mask_right_eye_r,0)[0][2])
            this_bottom = int(torch.max(mask_right_eye_r,0)[0][1])
            this_right = int(torch.max(mask_right_eye_r,0)[0][2])
            x_mean = int((this_left+this_right)/2)
            y_mean = int((this_top+this_bottom)/2)
            mask_tensor[2] = y_mean
            mask_tensor[3] = x_mean
            # mask_list.append(x_mean)
            # mask_list.append(y_mean)
        except:
            print("right eye problem --------------")
            print(A_path)
            mask_tensor[2] = 116
            mask_tensor[3] = 160
            # mask_list.append(116)
            # mask_list.append(160)

        try:
            mask_mouth_r = torch.nonzero((A_tensor==7)+(A_tensor==8)+(A_tensor==9))
            this_top = int(torch.min(mask_mouth_r,0)[0][1])
            this_left = int(torch.min(mask_mouth_r,0)[0][2])
            this_bottom = int(torch.max(mask_mouth_r,0)[0][1])
            this_right = int(torch.max(mask_mouth_r,0)[0][2])
            x_mean = int((this_left+this_right)/2)
            y_mean = int((this_top+this_bottom)/2)
            mask_tensor[4] = y_mean
            mask_tensor[5] = x_mean
        except:
            print("mouth problem --------------")
            print(A_path)
            mask_tensor[4] = 184
            mask_tensor[5] = 128
            # mask_list.append(184) # or 180
            # mask_list.append(128)

        assert 16<mask_tensor[0]<256-16
        assert 24<mask_tensor[1]<256-24
        assert 16<mask_tensor[2]<256-16
        assert 24<mask_tensor[3]<256-24
        assert 40<mask_tensor[4]<256-40
        assert 72<mask_tensor[5]<256-72

        # A_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A_tensor) * 255.0

        inst_tensor = feat_tensor = 0
        A_tensor = self.append_region(A,A_tensor,mask_tensor)


# ---------------------------------------------------------------------------------------------------------------

        mask_A_path = self.mask_A_paths[index]
        mask_A = Image.open(mask_A_path)        
        # params = get_params(self.opt, mask_A.size)

        mask_A_tensor = transforms.functional.to_tensor(mask_A) * 255.0

        w,h = mask_A.size
        max_size = max(w,h)

        if self.opt.longSize != max_size:
            scale_size = float(self.opt.longSize/max_size)
            new_w = int(scale_size * w)
            new_h = int(scale_size * h)
            mask_A = mask_A.resize((new_w,new_h),Image.NEAREST)
            # if self.opt.isTrain or self.opt.random_embed==False:
            mask_B_path = self.mask_B_paths[index]
            mask_B = Image.open(mask_B_path).convert('RGB')
            mask_B = mask_B.resize((new_w,new_h),Image.BICUBIC)
        else:
            # if self.opt.isTrain or self.opt.random_embed==False:
            mask_B_path = self.mask_B_paths[index]
            mask_B = Image.open(mask_B_path).convert('RGB')

        mask_A_tensor = transforms.functional.to_tensor(mask_A) * 255.0
        mask_B_tensor = transforms.functional.to_tensor(mask_B)
        real_mask_B_tensor = mask_B_tensor.clone()
        mask_bg = (mask_A_tensor==0).type(torch.FloatTensor)
        mask_B_tensor = torch.clamp(mask_B_tensor + mask_bg*torch.ones(mask_A_tensor.size()),0,1)
        mask_B = transforms.functional.to_pil_image(mask_B_tensor)                                      


        if self.opt.data_augmentation == True:
            assert self.opt.isTrain == True
            rotate,scale,shear = random.random()-0.5, random.random()-0.5, random.random()-0.5
            rotate,scale,shear = 0,0,0
            mask_B = transforms.functional.affine(mask_B, 20*rotate,[0,0],1+0.2*scale,10*shear,resample=Image.BICUBIC)
            mask_A = transforms.functional.affine(mask_A, 20*rotate,[0,0],1+0.2*scale,10*shear,resample=Image.NEAREST)
        

        # if self.opt.isTrain or self.opt.random_embed==False:
        mask_B_tensor = transforms.functional.to_tensor(mask_B)
        mask_B_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(mask_B_tensor)
        real_mask_B_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(real_mask_B_tensor)

        mask_A_tensor = transforms.functional.to_tensor(mask_A) * 255.0

        mask_tensor2 = torch.zeros(6)
        try:
            mask_left_eye_r = torch.nonzero(mask_A_tensor==4)
            this_top = int(torch.min(mask_left_eye_r,0)[0][1])
            this_left = int(torch.min(mask_left_eye_r,0)[0][2])
            this_bottom = int(torch.max(mask_left_eye_r,0)[0][1])
            this_right = int(torch.max(mask_left_eye_r,0)[0][2])
            x_mean = int((this_left+this_right)/2)
            y_mean = int((this_top+this_bottom)/2)
            mask_tensor2[0] = y_mean
            mask_tensor2[1] = x_mean
            # mask_list.append(x_mean)
            # mask_list.append(y_mean)
        except:
            print("left eye problem ------------------")
            print(mask_A_path)
            mask_tensor2[0] = 116
            mask_tensor2[1] = 96
            # mask_list.append(116)
            # mask_list.append(96)

        try:
            mask_right_eye_r = torch.nonzero(mask_A_tensor==5)
            this_top = int(torch.min(mask_right_eye_r,0)[0][1])
            this_left = int(torch.min(mask_right_eye_r,0)[0][2])
            this_bottom = int(torch.max(mask_right_eye_r,0)[0][1])
            this_right = int(torch.max(mask_right_eye_r,0)[0][2])
            x_mean = int((this_left+this_right)/2)
            y_mean = int((this_top+this_bottom)/2)
            mask_tensor2[2] = y_mean
            mask_tensor2[3] = x_mean
            # mask_list.append(x_mean)
            # mask_list.append(y_mean)
        except:
            print("right eye problem --------------")
            print(mask_A_path)
            mask_tensor2[2] = 116
            mask_tensor2[3] = 160
            # mask_list.append(116)
            # mask_list.append(160)

        try:
            mask_mouth_r = torch.nonzero((mask_A_tensor==7)+(mask_A_tensor==8)+(mask_A_tensor==9))
            this_top = int(torch.min(mask_mouth_r,0)[0][1])
            this_left = int(torch.min(mask_mouth_r,0)[0][2])
            this_bottom = int(torch.max(mask_mouth_r,0)[0][1])
            this_right = int(torch.max(mask_mouth_r,0)[0][2])
            x_mean = int((this_left+this_right)/2)
            y_mean = int((this_top+this_bottom)/2)
            mask_tensor2[4] = y_mean
            mask_tensor2[5] = x_mean
        except:
            print("mouth problem --------------")
            print(mask_A_path)
            mask_tensor2[4] = 184
            mask_tensor2[5] = 128
            # mask_list.append(184) # or 180
            # mask_list.append(128)

        assert 16<mask_tensor2[0]<256-16
        assert 24<mask_tensor2[1]<256-24
        assert 16<mask_tensor2[2]<256-16
        assert 24<mask_tensor2[3]<256-24
        assert 40<mask_tensor2[4]<256-40
        assert 72<mask_tensor2[5]<256-72

        mask_A_tensor = self.append_region(mask_A,mask_A_tensor,mask_tensor2)



        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'mask2': mask_tensor2, 'bg_styleimage': real_B_tensor, 'bg_contentimage': real_mask_B_tensor,
                      'feat': feat_tensor, 'path': A_path, 'image_affine': C_tensor, 'mask': mask_tensor, 'label2': mask_A_tensor}

        # content image:  bg_contentimage, label2, mask2
        # style image:  bg_styleimage, label, mask,       label,image_affine


        return input_dict

    def append_region(self,label,face_label,mask_tensor):
        w,h = label.size    
        new_w = int(1.1 * w)
        new_h = int(1.1 * h)
        label_scale = label.resize((new_w,new_h),Image.NEAREST)

        label_scale_tensor = transforms.functional.to_tensor(label_scale) * 255.0
        mask_tensor_scale = torch.zeros(6)        
        mask_tensor_diff = torch.zeros(6)
        for index in range(6):
            mask_tensor_scale[index] = int(1.1*mask_tensor[index])
            mask_tensor_diff[index] = int(mask_tensor_scale[index]-mask_tensor[index])

        # left_eye = label_scale.crop((mask_tensor_diff[0],mask_tensor_diff[1],mask_tensor_diff[0]+w,mask_tensor_diff[1]+h))
        # right_eye = label_scale.crop((mask_tensor_diff[2],mask_tensor_diff[3],mask_tensor_diff[2]+w,mask_tensor_diff[3]+h))
        # mouth = label_scale.crop((mask_tensor_diff[4],mask_tensor_diff[5],mask_tensor_diff[4]+w,mask_tensor_diff[5]+h))

        left_eye_mask_whole = label_scale_tensor[:,int(mask_tensor_diff[0]):int(mask_tensor_diff[0])+h,int(mask_tensor_diff[1]):int(mask_tensor_diff[1])+w]
        right_eye_mask_whole = label_scale_tensor[:,int(mask_tensor_diff[2]):int(mask_tensor_diff[2])+h,int(mask_tensor_diff[3]):int(mask_tensor_diff[3])+w]
        mouth_mask_whole = label_scale_tensor[:,int(mask_tensor_diff[4]):int(mask_tensor_diff[4])+h,int(mask_tensor_diff[5]):int(mask_tensor_diff[5])+w]

        left_eye_mask = (left_eye_mask_whole==4).type(torch.FloatTensor)
        right_eye_mask = (right_eye_mask_whole==5).type(torch.FloatTensor)
        mouth_mask = ((mouth_mask_whole==7)+(mouth_mask_whole==8)+(mouth_mask_whole==9)).type(torch.FloatTensor)

        face_label = left_eye_mask*left_eye_mask_whole + (1-left_eye_mask)*face_label
        face_label = right_eye_mask*right_eye_mask_whole + (1-right_eye_mask)*face_label
        face_label = mouth_mask*mouth_mask_whole + (1-mouth_mask)*face_label

        return face_label

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
