### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# append mouth and eye region

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
        params = get_params(self.opt, A.size)

        ### input B (real images)
        # if self.opt.isTrain or self.opt.random_embed==False:
        #     B_path = self.B_paths[index]
        #     B = Image.open(B_path).convert('RGB')
        #     transform_B = get_transform(self.opt, params)      
        #     B_tensor = transform_B(B)
        # else:
        #     B_tensor = 0

        # w,h = A.size
        # max_size = max(w,h)
        # if self.opt.longSize != max_size:
        #     scale_size = float(self.opt.longSize/max_size)
        #     new_w = int(scale_size * w)
        #     new_h = int(scale_size * h)
        #     A = A.resize((new_w,new_h),Image.NEAREST)
            
        #     if self.opt.isTrain or self.opt.random_embed==False:
        #         B = B.resize((new_w,new_h),Image.NEAREST)
        #         B_tensor = transform_B(B)
        #     else:
        #         B_tensor = 0


        # if self.opt.label_nc == 0:
        #     transform_A = get_transform(self.opt, params)
        #     A_tensor = transform_A(A.convert('RGB'))
        # else:
        #     transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        #     A_tensor = transform_A(A) * 255.0

        w,h = A.size
        max_size = max(w,h)

        # if self.opt.longSize != max_size:
        #     scale_size = float(self.opt.longSize/max_size)
        #     new_w = int(scale_size * w)
        #     new_h = int(scale_size * h)
        #     A = A.resize((new_w,new_h),Image.NEAREST)
        #     # if self.opt.isTrain or self.opt.random_embed==False:
        #     B_path = self.B_paths[index]
        #     B = Image.open(B_path).convert('RGB')
        #     B = B.resize((new_w,new_h),Image.BICUBIC)
        # else:
        #     # if self.opt.isTrain or self.opt.random_embed==False:
        #     B_path = self.B_paths[index]
        #     B = Image.open(B_path).convert('RGB')

        scale_size = float(self.opt.longSize/max_size)
        new_w = int(scale_size * w)
        new_h = int(scale_size * h)
        A = A.resize((new_w,new_h),Image.NEAREST)
        # if self.opt.isTrain or self.opt.random_embed==False:
        B_path = self.B_paths[index]
        B = Image.open(B_path).convert('RGB')
        B = B.resize((new_w,new_h),Image.BICUBIC)


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
            # if self.opt.isTrain or self.opt.random_embed==False:
            # B = transforms.functional.adjust_gamma(B, 0.7+1.2*random.random())
            # B = transforms.functional.adjust_hue(B, 0.04*(random.random()-0.5))
            # this is original image
            # B = transforms.functional.affine(B, 20*rotate,[0,0],1+0.2*scale,10*shear,resample=Image.BICUBIC)
            # A = transforms.functional.affine(A,20*rotate,[0,0],1+0.2*scale,10*shear,resample=Image.NEAREST)
            # this is larger affine noise
            B = transforms.functional.affine(B, 20*rotate,[0,0],1+0.2*scale,10*shear,resample=Image.BICUBIC)
            A = transforms.functional.affine(A, 20*rotate,[0,0],1+0.2*scale,10*shear,resample=Image.NEAREST)
            # C = transforms.functional.affine(B, 10*rotate,[0,0],1+0.1*scale,5*shear,resample=Image.BICUBIC)
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
        region_tensor = torch.zeros(12)
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
            # region_tensor[0] = this_left
            # region_tensor[1] = this_right
            # region_tensor[2] = this_top
            # region_tensor[3] = this_bottom
            # mask_list.append(x_mean)
            # mask_list.append(y_mean)
        except:
            print("left eye problem ------------------")
            print(A_path)
            mask_tensor[0] = 116
            mask_tensor[1] = 96
            # region_tensor[0] = 72
            # region_tensor[1] = 120
            # region_tensor[2] = 100
            # region_tensor[3] = 132
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
            # region_tensor[4] = this_left
            # region_tensor[5] = this_right
            # region_tensor[6] = this_top
            # region_tensor[7] = this_bottom
            # mask_list.append(x_mean)
            # mask_list.append(y_mean)
        except:
            print("right eye problem --------------")
            print(A_path)
            mask_tensor[2] = 116
            mask_tensor[3] = 160
            # region_tensor[4] = 136
            # region_tensor[5] = 184
            # region_tensor[6] = 100
            # region_tensor[7] = 132
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
            # region_tensor[8] = this_left
            # region_tensor[9] = this_right
            # region_tensor[10] = this_top
            # region_tensor[11] = this_bottom
        except:
            print("mouth problem --------------")
            print(A_path)
            mask_tensor[4] = 184
            mask_tensor[5] = 128
            # region_tensor[8] = 56
            # region_tensor[9] = 200
            # region_tensor[10] = 144
            # region_tensor[11] = 224
            # mask_list.append(184) # or 180
            # mask_list.append(128)

        assert 16<mask_tensor[0]<256-16
        assert 24<mask_tensor[1]<256-24
        assert 16<mask_tensor[2]<256-16
        assert 24<mask_tensor[3]<256-24
        assert 40<mask_tensor[4]<256-40
        assert 72<mask_tensor[5]<256-72

        # A_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A_tensor) * 255.0

        append_A_tensor = self.append_region(A,A_tensor,mask_tensor)

        inst_tensor = feat_tensor = 0

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': append_A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'bg_image': real_B_tensor, 'ori_label': A_tensor,
                      'feat': feat_tensor, 'path': A_path, 'image_affine': C_tensor, 'mask': mask_tensor}

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




        # mask_left_eye_image = torch.zeros(label.size()[0],3,32,48).cuda()
        # mask_right_eye_image = torch.zeros(label.size()[0],3,32,48).cuda()
        # mask_mouth_image = torch.zeros(label.size()[0],3,80,144).cuda()

        # mask_left_eye_image = face_label[:,int(region_tensor[2]):int(region_tensor[3]),int(region_tensor[0]):int(region_tensor[1])]
        # mask_right_eye_image = face_label[:,int(region_tensor[6]):int(region_tensor[7]),int(region_tensor[4]):int(region_tensor[5])]
        # mask_mouth_image = face_label[:,int(region_tensor[10]):int(region_tensor[11]),int(region_tensor[8]):int(region_tensor[9])]

        # mask_left_eye_image = mask_left_eye_image * (mask_left_eye_image == 4).type(torch.cuda.FloatTensor)
        # mask_right_eye_image = mask_right_eye_image * (mask_right_eye_image == 5).type(torch.cuda.FloatTensor)
        # mask_mouth_image = mask_mouth_image * ((mask_mouth_image==7)+(mask_mouth_image==8)+(mask_mouth_image==9)).type(torch.cuda.FloatTensor)

        # mask_left_eye_image = transforms.functional.to_pil_image(mask_left_eye_image)
        # mask_right_eye_image = transforms.functional.to_pil_image(mask_right_eye_image)
        # mask_mouth_image = transforms.functional.to_pil_image(mask_mouth_image)
        
        # new_w = int(1.1*(region_tensor[1]-region_tensor[0]))
        # new_h = int(1.1*(region_tensor[3]-region_tensor[2]))
        # mask_left_eye_image = mask_left_eye_image.resize((new_w,new_h),Image.NEAREST)
        # new_w = int(1.1*(region_tensor[5]-region_tensor[4]))
        # new_h = int(1.1*(region_tensor[7]-region_tensor[6]))
        # mask_right_eye_image = mask_right_eye_image.resize((new_w,new_h),Image.NEAREST)
        # new_w = int(1.1*(region_tensor[9]-region_tensor[8]))
        # new_h = int(1.1*(region_tensor[11]-region_tensor[10]))
        # mask_mouth_image = mask_mouth_image.resize((new_w,new_h),Image.NEAREST)

        # mask_left_eye_image = transforms.functional.to_tensor(mask_left_eye_image)
        # mask_right_eye_image = transforms.functional.to_tensor(mask_right_eye_image)
        # mask_mouth_image = transforms.functional.to_tensor(mask_mouth_image)
