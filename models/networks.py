### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from . import bninception
import os
# 256 and 512
###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):   
    ### by default netG is global 
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def define_embed_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', embed_nc=128, gpu_ids=[]):
    ### by default netG is global, and we only implement global netG at the first time
    if netG == "global":
        netG = EmbedGlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm, embed_nc)
    else:
        raise('embed_G has only implemented global embedG')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG    

def define_embed_bg_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', embed_nc=128, gpu_ids=[]):
    ### by default netG is global, and we only implement global netG at the first time
    if netG == "global":
        netG = EmbedGlobalBGGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm, embed_nc)
    else:
        raise('embed_G has only implemented global embedG')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

class EmbedGlobalBGGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm="batch", embed_nc=128,
                 padding_type='reflect'):
        norm_layer = get_norm_layer(norm_type=norm)
        assert(n_blocks >= 0)
        super(EmbedGlobalBGGenerator, self).__init__()
        activation = nn.ReLU(True)        

        downsample_model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            if i != n_downsampling-1:
                downsample_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
            else:
                downsample_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        self.downsample_model = nn.Sequential(*downsample_model)

        ### resnet blocks
        model=[]
        model += [nn.Conv2d(in_channels=ngf*(2**n_downsampling)+embed_nc, out_channels=ngf*(2**n_downsampling), kernel_size=1, padding=0, stride=1, bias=True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0),
                       norm_layer(int(ngf * mult / 2)), activation]
        # model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
        # self.compress_channel = nn.Sequential(
        #     nn.Conv2d(in_channels=ngf*(2**n_downsampling)+embed_nc, out_channels=ngf*(2**n_downsampling), kernel_size=1, padding=0, stride=1, bias=False))

        #define background encoder model
        bg_encoder = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        self.bg_encoder = nn.Sequential(*bg_encoder)

        bg_decoder = [nn.Conv2d(in_channels=ngf*2, out_channels=ngf, kernel_size=1, padding=0, stride=1, bias=True)]
        bg_decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.bg_decoder = nn.Sequential(*bg_decoder)


    def forward(self, input, type="label_encoder"):
        if type=="label_encoder":
            return self.downsample_model(input)
        elif type=="image_G":
            return self.model(input)
        elif type=="bg_encoder":
            return self.bg_encoder(input)
        elif type=="bg_decoder":
            # notice before bg_decoder, we should concate the feature map form G and bg_encoder
            return self.bg_decoder(input)
        else:
            print("wrong type ! ")

class EmbedGlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm="batch", embed_nc=128,
                 padding_type='reflect'):
        norm_layer = get_norm_layer(norm_type=norm)
        assert(n_blocks >= 0)
        super(EmbedGlobalGenerator, self).__init__()
        activation = nn.ReLU(True)        

        downsample_model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            if i != n_downsampling-1:
                downsample_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
            else:
                downsample_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        self.downsample_model = nn.Sequential(*downsample_model)

        ### resnet blocks
        model=[]
        model += [nn.Conv2d(in_channels=ngf*(2**n_downsampling)+embed_nc, out_channels=ngf*(2**n_downsampling), kernel_size=1, padding=0, stride=1, bias=True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
        # self.compress_channel = nn.Sequential(
        #     nn.Conv2d(in_channels=ngf*(2**n_downsampling)+embed_nc, out_channels=ngf*(2**n_downsampling), kernel_size=1, padding=0, stride=1, bias=False))

    def forward(self, input, type="label_encoder"):
        if type=="label_encoder":
            return self.downsample_model(input)
        elif type=="image_G":
            return self.model(input)
        else:
            print("wrong type ! ")
            
# use longsize to select which network
def define_decoder_mask(longsize=512, norm='instance',gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if longsize == 256:
        net_decoder = DecoderGenerator_mask_skin(norm_layer)  # input longsize 256 to 128*4*4
    elif longsize == 80:
        net_decoder = DecoderGenerator_mask_mouth(norm_layer)  # input longsize 256 to 512*4*4
    elif longsize == 32:
        net_decoder = DecoderGenerator_mask_eye(norm_layer)  # input longsize 256 to 512*4*4
    else:
        print("not implemented !!")

    print("net_decoder of size "+str(longsize)+" is:")
    print(net_decoder)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net_decoder.cuda(gpu_ids[0])
    net_decoder.apply(weights_init)
    return net_decoder

# use longsize to select which network
def define_decoder_mask_image(longsize=512, norm='instance',gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if longsize == 256:
        net_decoder = DecoderGenerator_mask_skin_image(norm_layer)  # input longsize 256 to 128*4*4
    elif longsize == 80:
        net_decoder = DecoderGenerator_mask_mouth_image(norm_layer)  # input longsize 256 to 512*4*4
    elif longsize == 32:
        net_decoder = DecoderGenerator_mask_eye_image(norm_layer)  # input longsize 256 to 512*4*4
    else:
        print("not implemented !!")

    print("net_decoder to image of size "+str(longsize)+" is:")
    print(net_decoder)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net_decoder.cuda(gpu_ids[0])
    net_decoder.apply(weights_init)
    return net_decoder

class DecoderGenerator_mask_skin_image(nn.Module):
    def __init__(self, norm_layer):  
        super(DecoderGenerator_mask_skin_image, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512*2*2))
        # input is 512*2*2
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2, output_padding=0))  #128*4
        layers_list.append(DecoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2, output_padding=0))  #128*8*8
        layers_list.append(DecoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2, output_padding=0))  #128*16*16
        layers_list.append(DecoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2, output_padding=0))  #128*32*32
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0))  #128*64*64
        layers_list.append(DecoderBlock(channel_in=256, channel_out=128, kernel_size=4, padding=1, stride=2, output_padding=0))  #64*128*128
        layers_list.append(DecoderBlock(channel_in=128, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0))  #64*256*256
        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(64,3,kernel_size=5,padding=0))
        layers_list.append(nn.Tanh())
        
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator_mask_skin, print some shape ")
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0],512, 2, 2)
        ten = self.conv(ten)
        assert ten.size()[1] == 3
        assert ten.size()[2] == 256
        assert ten.size()[3] == 256
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_skin_image, self).__call__(*args, **kwargs)

class DecoderGenerator_mask_mouth_image(nn.Module):
    def __init__(self, norm_layer):  
        super(DecoderGenerator_mask_mouth_image, self).__init__()
        # start from B*1024
        # self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=512*4*4),
        #                         nn.BatchNorm1d(num_features=512*4*4, momentum=0.9),
        #                         nn.ReLU(True))
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512*5*9))
        layers_list = []
        # layers_list.append(nn.BatchNorm2d(256, momentum=0.9))
        # layers_list.append(nn.ReLU(True))

        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0)) #10*18
        layers_list.append(DecoderBlock(channel_in=256, channel_out=128, kernel_size=4, padding=1, stride=2, output_padding=0)) #20*36
        layers_list.append(DecoderBlock(channel_in=128, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #40*72
        layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #80*144
        # layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #96*160
        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(64,3,kernel_size=5,padding=0))
        layers_list.append(nn.Tanh())

        # layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=3, padding=1, stride=1, output_padding=0)) #256*12*14

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator, print some shape ")
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0],512, 5, 9)
        ten = self.conv(ten)
        assert ten.size()[1] == 3
        assert ten.size()[2] == 80
        assert ten.size()[3] == 144
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_mouth_image, self).__call__(*args, **kwargs)


class DecoderGenerator_mask_eye_image(nn.Module):
    def __init__(self, norm_layer):  
        super(DecoderGenerator_mask_eye_image, self).__init__()
        # start from B*1024
        # self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=512*4*4),
        #                         nn.BatchNorm1d(num_features=512*4*4, momentum=0.9),
        #                         nn.ReLU(True))
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512*2*3, bias=False))
        layers_list = []
        # layers_list.append(nn.BatchNorm2d(256, momentum=0.9))
        # layers_list.append(nn.ReLU(True))

        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0)) #256*4
        layers_list.append(DecoderBlock(channel_in=256, channel_out=128, kernel_size=4, padding=1, stride=2, output_padding=0)) #128*8
        layers_list.append(DecoderBlock(channel_in=128, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #64*16
        layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #64*32
        # layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #64*64
        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(64,3,kernel_size=5,padding=0))
        layers_list.append(nn.Tanh())

        # layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=3, padding=1, stride=1, output_padding=0)) #256*12*14

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator, print some shape ")
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0],512, 2, 3)
        ten = self.conv(ten)
        assert ten.size()[1] == 3
        assert ten.size()[2] == 32
        assert ten.size()[3] == 48
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_eye_image, self).__call__(*args, **kwargs)




class DecoderGenerator_mask_skin(nn.Module):
    def __init__(self, norm_layer):  
        super(DecoderGenerator_mask_skin, self).__init__()
        # input is 128*4*4
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512*2*2))
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2))  #256*4
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2))  #256*8
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2))  #256*16
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2))  #256*32
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2))  #256*64
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator_mask_skin, print some shape ")
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0],512, 2, 2)
        ten = self.conv(ten)
        assert ten.size()[1] == 256
        assert ten.size()[2] == 64
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_skin, self).__call__(*args, **kwargs)


class DecoderGenerator_mask160(nn.Module):
    def __init__(self, norm_layer):  
        super(DecoderGenerator_mask160, self).__init__()
        # start from B*1024
        # self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=256*5*5, bias=False),
        #                         nn.BatchNorm1d(num_features=256*5*5, momentum=0.9),
        #                         nn.ReLU(True))
        # self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=256*5*5, bias=False))

        layers_list = []
        # layers_list.append(nn.BatchNorm2d(256, momentum=0.9))
        # layers_list.append(nn.ReLU(True))
        #512*2*2
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0)) #256*4
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=1)) #256*9
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0)) #256*18
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=3, padding=1, stride=1, output_padding=0)) #256*18
        # layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0)) #256*36
        # layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=3, padding=1, stride=1, output_padding=0)) #256*72

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator, print some shape ")
        ten = self.conv(ten)
        assert ten.size()[1] == 256
        assert ten.size()[2] == 18
        assert ten.size()[3] == 18
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask160, self).__call__(*args, **kwargs)

class DecoderGenerator_mask_mouth(nn.Module):
    def __init__(self, norm_layer):  
        super(DecoderGenerator_mask_mouth, self).__init__()
        

        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512*5*9))
        layers_list = []

        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2)) #10*18
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2)) #20*36
        # layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2)) #40*72

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator, print some shape ")
        # ten = self.fc(ten)
        # ten = ten.view(ten.size()[0],512, 4, 4)
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0],512, 5, 9)
        ten = self.conv(ten)
        assert ten.size()[1] == 256
        assert ten.size()[2] == 20
        assert ten.size()[3] == 36
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_mouth, self).__call__(*args, **kwargs)


class DecoderGenerator_mask_eye(nn.Module):
    def __init__(self, norm_layer):  
        super(DecoderGenerator_mask_eye, self).__init__()
        # start from B*1024
        # self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=512*4*4),
        #                         nn.BatchNorm1d(num_features=512*4*4, momentum=0.9),
        #                         nn.ReLU(True))
        # self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=256*6*7, bias=False))
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512*2*3, bias=False))
        layers_list = []
        # layers_list.append(nn.BatchNorm2d(256, momentum=0.9))
        # layers_list.append(nn.ReLU(True))
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2)) #256*4
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2)) #256*8
        # layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2)) #256*16
        # layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=3, padding=1, stride=1)) #256*16
        # # layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=3, padding=1, stride=1, output_padding=0)) #256*12*14

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator, print some shape ")
        # ten = self.fc(ten)
        # ten = ten.view(ten.size()[0],512, 4, 4)
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0],512, 2, 3)
        ten = self.conv(ten)
        assert ten.size()[1] == 256
        assert ten.size()[2] == 8
        assert ten.size()[3] == 12
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_eye, self).__call__(*args, **kwargs)


# use longsize to select which network
def define_encoder_mask(longsize=512, norm='instance',gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if longsize == 256:
        net_encoder = EncoderGenerator_mask_skin(norm_layer)  # input longsize 256 to 128*4*4
    elif longsize == 80:
        net_encoder = EncoderGenerator_mask_mouth(norm_layer)  # input longsize 256 to 512*4*4
    elif longsize == 32:
        net_encoder = EncoderGenerator_mask_eye(norm_layer)  # input longsize 256 to 512*4*4
    else:
        print("not implemented !!")

    print("net_encoder of size "+str(longsize)+" is:")
    print(net_encoder)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net_encoder.cuda(gpu_ids[0])
    net_encoder.apply(weights_init)
    return net_encoder

class  EncoderGenerator_mask_mouth(nn.Module):
    """docstring for  EncoderGenerator"""
    def __init__(self, norm_layer):
        super( EncoderGenerator_mask_mouth, self).__init__()
        layers_list = []
        
        # 3*80*144
        layers_list.append(EncoderBlock(channel_in=3, channel_out=64, kernel_size=4, padding=1, stride=2))  # 40*72
        layers_list.append(EncoderBlock(channel_in=64, channel_out=128, kernel_size=4, padding=1, stride=2))  # 20*36
        layers_list.append(EncoderBlock(channel_in=128, channel_out=256, kernel_size=4, padding=1, stride=2))  # 10*18
        layers_list.append(EncoderBlock(channel_in=256, channel_out=512, kernel_size=4, padding=1, stride=2))  # 5*9
        # layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2))  # 3*5
        
        # final shape Bx256*7*6
        self.conv = nn.Sequential(*layers_list)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=512*5*9, out_features=1024),
                                # nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.ReLU(True),
                                nn.Linear(in_features=1024, out_features=512))
        self.fc_var = nn.Sequential(nn.Linear(in_features=512*5*9, out_features=1024),
                                # nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.ReLU(True),
                                nn.Linear(in_features=1024, out_features=512))
        # self.c_mu = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)
        # self.c_var = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)


    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(ten.size()[0],-1)
        mu = self.fc_mu(ten)
        logvar = self.fc_var(ten)
        return mu,logvar

    def __call__(self, *args, **kwargs):
        return super(EncoderGenerator_mask_mouth, self).__call__(*args, **kwargs)



class  EncoderGenerator_mask_eye(nn.Module):
    """docstring for  EncoderGenerator"""
    def __init__(self, norm_layer):
        super( EncoderGenerator_mask_eye, self).__init__()
        layers_list = []
        
        # 3*32*48
        layers_list.append(EncoderBlock(channel_in=3, channel_out=64, kernel_size=4, padding=1, stride=2))  # 16*24
        layers_list.append(EncoderBlock(channel_in=64, channel_out=128, kernel_size=4, padding=1, stride=2))  # 
        layers_list.append(EncoderBlock(channel_in=128, channel_out=256, kernel_size=4, padding=1, stride=2))  # 4*6
        layers_list.append(EncoderBlock(channel_in=256, channel_out=512, kernel_size=4, padding=1, stride=2))  # 512*2*3
        
        # final shape Bx256*7*6
        self.conv = nn.Sequential(*layers_list)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=512*2*3, out_features=1024),
                                # nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.ReLU(True),
                                nn.Linear(in_features=1024, out_features=512))
        self.fc_var = nn.Sequential(nn.Linear(in_features=512*2*3, out_features=1024),
                                # nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.ReLU(True),
                                nn.Linear(in_features=1024, out_features=512))
        # self.c_mu = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)
        # self.c_var = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)


    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(ten.size()[0],-1)
        mu = self.fc_mu(ten)
        logvar = self.fc_var(ten)
        return mu,logvar

    def __call__(self, *args, **kwargs):
        return super(EncoderGenerator_mask_eye, self).__call__(*args, **kwargs)




class  EncoderGenerator_mask_skin(nn.Module):
    """docstring for  EncoderGenerator"""
    def __init__(self, norm_layer):
        super( EncoderGenerator_mask_skin, self).__init__()
        layers_list = []
        
        # 3*256*256
        layers_list.append(EncoderBlock(channel_in=3, channel_out=64, kernel_size=4, padding=1, stride=2))  # 64*128*128
        layers_list.append(EncoderBlock(channel_in=64, channel_out=128, kernel_size=4, padding=1, stride=2))  # 128*64*64
        layers_list.append(EncoderBlock(channel_in=128, channel_out=256, kernel_size=4, padding=1, stride=2))  # 128*32*32
        layers_list.append(EncoderBlock(channel_in=256, channel_out=512, kernel_size=4, padding=1, stride=2))  # 128*16*16
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2))  # 128*8*8
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2))  # 512*4*4
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2))  # 512*2*2
        # final shape Bx128*4*4
        self.conv = nn.Sequential(*layers_list)

        # self.c_mu = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1)
        # self.c_var = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=512*2*2, out_features=1024),
                                # nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.ReLU(True),
                                nn.Linear(in_features=1024, out_features=512))
        self.fc_var = nn.Sequential(nn.Linear(in_features=512*2*2, out_features=1024),
                                # nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.ReLU(True),
                                nn.Linear(in_features=1024, out_features=512))

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(ten.size()[0],-1)
        mu = self.fc_mu(ten)
        logvar = self.fc_var(ten)
        return mu,logvar

    def __call__(self, *args, **kwargs):
        return super(EncoderGenerator_mask_skin, self).__call__(*args, **kwargs)





def define_encoder(n_downsample_global, input_nc,embed_length,norm='instance',gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net_encoder = EncoderGenerator(n_downsample_global, input_nc,embed_length,norm_layer)
    print("net_encoder ")
    print(net_encoder)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net_encoder.cuda(gpu_ids[0])
    net_encoder.apply(weights_init)
    return net_encoder

def define_encoder_vector(n_downsample_global, input_nc,embed_length,norm='instance',gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net_encoder = EncoderVectorGenerator(n_downsample_global, input_nc,embed_length,norm_layer)
    print("net_encoder vector ")
    print(net_encoder)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net_encoder.cuda(gpu_ids[0])
    net_encoder.apply(weights_init)
    return net_encoder


def define_decoder(n_downsample_global, embed_length, embed_feature_size, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net_decoder = DecoderGenerator(embed_length, embed_feature_size, n_downsample_global, norm_layer)
    print(net_decoder)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net_decoder.cuda(gpu_ids[0])
    net_decoder.apply(weights_init)
    return net_decoder

class DecoderGenerator_512_64(nn.Module):
    def __init__(self, norm_layer):  
        super(DecoderGenerator_512_64, self).__init__()
        # input is 512*4*4
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=512, channel_out=512))  #512*8*8
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256))  #256*16*16
        layers_list.append(DecoderBlock(channel_in=256, channel_out=128))  #128*32*32
        layers_list.append(DecoderBlock(channel_in=128, channel_out=64))   #64*64*64
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator, print some shape ")
        ten = self.conv(ten)
        assert ten.size()[1] == 64
        assert ten.size()[2] == 64
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_512_64, self).__call__(*args, **kwargs)



class DecoderGenerator(nn.Module):
    def __init__(self, embed_length, output_size, n_downsample_global, norm_layer):  #output_size is output feature map size decided by image size and n_downsample_layer
        # by default output_size = 512/8=64, output_nc = 128
        output_nc = 2**(n_downsample_global+4)
        print("in DecoderGenerator ")
        print(output_size)      #64   #32    #64
        print(output_nc)        #128  #256   #64
        super(DecoderGenerator, self).__init__()
        # start from B*z_size
        self.fc = nn.Sequential(nn.Linear(in_features=embed_length, out_features=8 * 8 * output_nc//4, bias=False),
                                nn.BatchNorm1d(num_features=8 * 8 * output_nc//4, momentum=0.9),
                                # nn.InstanceNorm1d(num_features=8 * 8 * output_nc//4, momentum=0.9,track_running_stats=True),
                                nn.ReLU(True))
        self.output_nc = output_nc
        self.output_size = output_size
        layers_list = []
        if n_downsample_global == 3:                
            layers_list.append(DecoderBlock(channel_in=self.output_nc//4, channel_out=self.output_nc//2))
            layers_list.append(DecoderBlock(channel_in=self.output_nc//2, channel_out=self.output_nc))
            layers_list.append(DecoderBlock(channel_in=self.output_nc, channel_out=self.output_nc))
        elif n_downsample_global == 4:
            layers_list.append(DecoderBlock(channel_in=self.output_nc//4, channel_out=self.output_nc//2))
            layers_list.append(DecoderBlock(channel_in=self.output_nc//2, channel_out=self.output_nc))
            # layers_list.append(DecoderBlock(channel_in=self.output_nc, channel_out=self.output_nc))
        elif n_downsample_global == 2:
            layers_list.append(DecoderBlock(channel_in=self.output_nc//4, channel_out=self.output_nc//2))
            layers_list.append(DecoderBlock(channel_in=self.output_nc//2, channel_out=self.output_nc))
            layers_list.append(DecoderBlock(channel_in=self.output_nc, channel_out=self.output_nc))
        else:
            print("n_downsample_global error")


        # self.size = self.size//4
        # final conv to get 3 channels and tanh layer
        # layers_list.append(nn.Sequential(
        #     nn.Conv2d(in_channels=self.output_nc, out_channels=self.output_nc, kernel_size=5, stride=1, padding=2),
        #     nn.Tanh()
        # ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator, print some shape ")
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0],-1, 8, 8)
        ten = self.conv(ten)
        assert ten.size()[1] == self.output_nc
        assert ten.size()[2] == self.output_size
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator, self).__call__(*args, **kwargs)

class  EncoderVectorGenerator(nn.Module):
    """docstring for  EncoderVectorGenerator"""
    def __init__(self, n_downsample_global,input_nc,embed_length,norm_layer):
        super( EncoderVectorGenerator, self).__init__()
        self.size = input_nc
        layers_list = []
        # the first time 3->64, for every other double the channel size
        for i in range(n_downsample_global):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=64))
                self.size = 64
            else:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
                self.size *= 2

        # final shape Bx256x8x8
        self.conv = nn.Sequential(*layers_list)
        if n_downsample_global == 2:
            feature_size = 16
        elif n_downsample_global == 3:
            feature_size = 8
        else:
            print("feature_size unknown")
        self.fc = nn.Sequential(nn.Linear(in_features=feature_size * feature_size * self.size, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                # nn.InstanceNorm1d(num_features=1024,momentum=0.9,track_running_stats=True),
                                nn.ReLU(True),
                                nn.Linear(in_features=1024, out_features=embed_length))
        # 16 for n_downsample_global is 2
        # 8 for n_downsample_global is 3
        # # two linear to get the mu vector and the diagonal of the log_variance

    def forward(self, ten):
        # print("in EncoderVectorGenerator, print some shape ")
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        output = self.fc(ten)  #1,1024
        # ten = torch.unsqueeze(ten,1)
        return output

    def __call__(self, *args, **kwargs):
        return super(EncoderVectorGenerator, self).__call__(*args, **kwargs)

class  EncoderGenerator(nn.Module):
    """docstring for  EncoderGenerator"""
    def __init__(self, n_downsample_global,input_nc,embed_length,norm_layer):
        super( EncoderGenerator, self).__init__()
        self.size = input_nc
        layers_list = []
        # the first time 3->64, for every other double the channel size
        for i in range(n_downsample_global):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=64))
                self.size = 64
            else:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
                self.size *= 2

        # final shape Bx256x8x8
        self.conv = nn.Sequential(*layers_list)
        if n_downsample_global == 2:
            feature_size = 16
        elif n_downsample_global == 3:
            feature_size = 8
        else:
            print("feature_size unknown")
        self.fc = nn.Sequential(nn.Linear(in_features=feature_size * feature_size * self.size, out_features=1024, bias=False))
        
        self.act = nn.Sequential(nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                # nn.InstanceNorm1d(num_features=1024,momentum=0.9,track_running_stats=True),
                                nn.ReLU(True))
        # 16 for n_downsample_global is 2
        # 8 for n_downsample_global is 3

        # # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=1024, out_features=embed_length)
        self.l_var = nn.Linear(in_features=1024, out_features=embed_length)

    def forward(self, ten):
        # print("in EncoderGenerator, print some shape ")
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)  #1,1024
        # ten = torch.unsqueeze(ten,1)
        ten = self.act(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu,logvar

    def __call__(self, *args, **kwargs):
        return super(EncoderGenerator, self).__call__(*args, **kwargs)

class  EncoderGenerator_256_512(nn.Module):
    """docstring for  EncoderGenerator"""
    def __init__(self, norm_layer):
        super( EncoderGenerator_256_512, self).__init__()
        layers_list = []
        
        # 3*256*256
        layers_list.append(EncoderBlock(channel_in=3, channel_out=64, kernel_size=5, padding=2, stride=2))  # 64*128*128
        layers_list.append(EncoderBlock(channel_in=64, channel_out=128, kernel_size=5, padding=2, stride=2))  # 128*64*64
        layers_list.append(EncoderBlock(channel_in=128, channel_out=256, kernel_size=3, padding=1, stride=2))  # 256*32*32
        layers_list.append(EncoderBlock(channel_in=256, channel_out=512, kernel_size=3, padding=1, stride=2))  # 512*16*16
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=3, padding=1, stride=2))  # 512*8*8
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=3, padding=1, stride=2))  # 512*4*4
        
        # final shape Bx512*8*8
        self.conv = nn.Sequential(*layers_list)
        
        self.c_mu = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.c_var = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)

    def forward(self, ten):
        # print("in EncoderGenerator, print some shape ")
        ten = self.conv(ten)
        mu = self.c_mu(ten)
        logvar = self.c_var(ten)
        return mu,logvar

    def __call__(self, *args, **kwargs):
        return super(EncoderGenerator_256_512, self).__call__(*args, **kwargs)



# # encoder block (used in encoder and discriminator)
# class EncoderBlock(nn.Module):
#     def __init__(self, channel_in, channel_out, kernel_size=7, padding=3, stride=4):
#         super(EncoderBlock, self).__init__()
#         # convolution to halve the dimensions
#         self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
#         self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)
#         # self.bn = nn.InstanceNorm2d(num_features=channel_out, momentum=0.9,track_running_stats=True)

#     def forward(self, ten, out=False,t = False):
#         # here we want to be able to take an intermediate output for reconstruction error
#         if out:
#             ten = self.conv(ten)
#             ten_out = ten
#             ten = self.bn(ten)
#             ten = nn.functional.relu(ten, False)
#             return ten, ten_out
#         else:
#             ten = self.conv(ten)
#             ten = self.bn(ten)
#             ten = nn.functional.relu(ten, True)
#             return ten

# encoder block (used in encoder and discriminator)
class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=7, padding=3, stride=4):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)
        self.relu = nn.ReLU(True)

    def forward(self, ten, out=False,t = False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = self.relu(ten)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = self.relu(ten)
            return ten

# encoder block (used in encoder and discriminator)
class EncoderResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=7, padding=3, stride=4):
        super(EncoderResBlock, self).__init__()
        # convolution to halve the dimensions
        layers_list1 = []
        layers_list1.append(nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, padding=1, stride=1))
        layers_list1.append(nn.BatchNorm2d(num_features=channel_out, momentum=0.9))
        layers_list1.append(nn.ReLU(True))
        layers_list1.append(nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3, padding=1, stride=1))
        layers_list1.append(nn.BatchNorm2d(num_features=channel_out, momentum=0.9))
        layers_list1.append(nn.ReLU(True))
        layers_list1.append(nn.AvgPool2d(2,2))
        self.conv1 = nn.Sequential(*layers_list1)

        layers_list2 = []
        layers_list2.append(nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, padding=0, stride=1))
        layers_list2.append(nn.AvgPool2d(2,2))
        self.conv2 = nn.Sequential(*layers_list2)

    def forward(self, ten):
        # here we want to be able to take an intermediate output for reconstruction error
        ten1 = self.conv1(ten)
        ten2 = self.conv2(ten)
        return ten1+ten2

# decoder block (used in the decoder)
class DecoderResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=4, padding=1, stride=2, output_padding=0, norelu=False):
        super(DecoderResBlock, self).__init__()
        
        layers_list1 = []
        layers_list1.append(nn.Upsample(scale_factor=2,mode='nearest'))
        layers_list1.append(nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, padding=1, stride=1))
        layers_list1.append(nn.BatchNorm2d(num_features=channel_out, momentum=0.9))
        layers_list1.append(nn.ReLU(True))
        layers_list1.append(nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3, padding=1, stride=1))
        layers_list1.append(nn.BatchNorm2d(num_features=channel_out, momentum=0.9))
        layers_list1.append(nn.ReLU(True))
        self.conv1 = nn.Sequential(*layers_list1)

        layers_list2 = []
        layers_list2.append(nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, padding=0, stride=1))
        layers_list2.append(nn.Upsample(scale_factor=2,mode='nearest'))
        self.conv2 = nn.Sequential(*layers_list2)        

    def forward(self, ten):
        ten1 = self.conv1(ten)
        ten2 = self.conv2(ten)
        return ten1+ten2



# decoder block (used in the decoder)
class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=4, padding=1, stride=2, output_padding=0, norelu=False):
        super(DecoderBlock, self).__init__()
        # transpose convolution to double the dimensions
        # self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding)
        # self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)
        # self.bn = nn.InstanceNorm2d(channel_out, momentum=0.9,track_running_stats=True)
        layers_list = []
        layers_list.append(nn.ConvTranspose2d(channel_in, channel_out, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding))
        layers_list.append(nn.BatchNorm2d(channel_out, momentum=0.9))
        if norelu == False:
            layers_list.append(nn.ReLU(True))
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.conv(ten)
        return ten




def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class ID_Loss(nn.Module):
    def __init__(self, gpu_ids, load_path):
        super(ID_Loss, self).__init__()        
        self.model = bninception.BNInception(num_classes=128)
        assert load_path != ''
        self.load_face_id_model(self.model, load_path)
        self.model = self.model.cuda()
        self.criterion = nn.L1Loss()

    def load_face_id_model(self, network, load_path):
        assert os.path.isfile(load_path) == True        
        try:
            network.load_state_dict(torch.load(load_path))
        except:   
            pretrained_dict = torch.load(load_path)
            model_dict = network.state_dict()
            pretrained_dict = {k.replace("module.backbone.",""): v for k, v in pretrained_dict.items() if k.replace("module.backbone.","") in model_dict}
            network.load_state_dict(pretrained_dict)

    def forward(self, x, y):
        x_128feature, y_128feature = self.model(x), self.model(y)
        loss = self.criterion(x_128feature, y_128feature.detach())
        return loss

# this is origianal vggloss without mask list
# class VGGLoss(nn.Module):
#     def __init__(self, gpu_ids, weights = None):
#         super(VGGLoss, self).__init__()       
#         if weights != None: 
#             self.weights = weights
#         else:
#             self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        
#         self.vgg = Vgg19().cuda()
#         self.criterion = nn.L1Loss()

#     def forward(self, x, y, face_mask, weights):              
#         if weights != None: 
#             self.weights = weights
#         mask = []
#         mask.append(face_mask)
#         x_vgg, y_vgg = self.vgg(x,layers_num=len(self.weights)), self.vgg(y,layers_num=len(self.weights))
#         downsample = nn.MaxPool2d(2)
#         for i in range(len(x_vgg)):
#             mask.append(downsample(mask[i]))
#             mask[i] = mask[i].detach()
#         loss = 0
#         for i in range(len(x_vgg)):
#             loss += self.weights[i] * self.criterion(x_vgg[i]*mask[i], (y_vgg[i]*mask[i]).detach())        
#         return loss



class VGGLoss(nn.Module):
    def __init__(self, gpu_ids, weights = None):
        super(VGGLoss, self).__init__()       
        if weights != None: 
            self.weights = weights
        else:
            self.weights = [1.0/4, 1.0/4, 1.0/4, 1.0/8, 1.0/8]        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()

    def forward(self, x, y, face_mask, mask_weights):              
        assert face_mask.size()[1] == len(mask_weights)  # suppose to be 5
        x_vgg, y_vgg = self.vgg(x,layers_num=len(self.weights)), self.vgg(y,layers_num=len(self.weights))
        mask = []
        mask.append(face_mask.detach())
        
        downsample = nn.MaxPool2d(2)
        for i in range(len(x_vgg)):
            mask.append(downsample(mask[i]))
            mask[i] = mask[i].detach()
        loss = 0
        for i in range(len(x_vgg)):
            for mask_index in range(len(mask_weights)):
                a = x_vgg[i]*mask[i][:,mask_index:mask_index+1,:,:]
                loss += self.weights[i] * self.criterion(x_vgg[i]*mask[i][:,mask_index:mask_index+1,:,:], (y_vgg[i]*mask[i][:,mask_index:mask_index+1,:,:]).detach()) * mask_weights[mask_index]
        return loss



class MFMLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(MFMLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x_input, y_input):
        loss = 0
        for i in range(len(x_input)):
            x = x_input[i][-2]
            y = y_input[i][-2]
            assert x.dim() == 4 
            assert y.dim() == 4
            x_mean = torch.mean(x,0)
            y_mean = torch.mean(y,0)
            loss += self.criterion(x_mean, y_mean.detach())
        return loss

def grammatrix(feature):
    assert feature.dim() == 4
    a,b,c,d = feature.size()[0],feature.size()[1],feature.size()[2],feature.size()[3]
    out_tensor = torch.Tensor(a,b,b).cuda()
    for batch_index in range(0,a):
        features = feature[batch_index].view(b,c*d)
        G=torch.mm(features,features.t())
        out_tensor[batch_index] = G.clone().div(b*c*d)
    return out_tensor

class GramMatrixLoss(nn.Module):
    def __init__(self,gpu_ids):
        super(GramMatrixLoss, self).__init__()        
        self.weights = [1.0,1.0,1.0]
        self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.MSELoss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y, label):
        # we use this label to label face
        face_mask = (label==1).type(torch.cuda.FloatTensor)
        mask = []
        mask.append(face_mask)
        x_vgg, y_vgg = self.vgg(x,layers_num=len(self.weights)), self.vgg(y,layers_num=len(self.weights))
        downsample = nn.MaxPool2d(2)
        for i in range(len(x_vgg)):
            mask.append(downsample(mask[i]))
            mask[i] = mask[i].detach()
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(grammatrix(x_vgg[i]*mask[i]), grammatrix(y_vgg[i]*mask[i]).detach())
        return loss



##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=4, stride=2, padding=1, output_padding=0), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)             
        
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[-2:]
        else:
            return [model(input)]

    def forward(self, input):        
        # print("do some debug in MultiscaleDiscriminator ***********************************")
        # print(self.getIntermFeat)
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            # print("i is ")
            # print(i)
            # print("input_downsampled size is ")
            # print(input_downsampled.size())
            
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            # res = [input]
            # for n in range(self.n_layers+2):
            #     model = getattr(self, 'model'+str(n))
            #     res.append(model(res[-1]))
            # return res[1:]
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            print("debug in networks line 721 ----")
            print(len(res[-2:]))
            return res[-2:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, layers_num=5):
        h_relu1 = self.slice1(X)
        if layers_num == 1:
            return [h_relu1]   
        h_relu2 = self.slice2(h_relu1)     
        if layers_num == 2:
            return [h_relu1, h_relu2]   
        h_relu3 = self.slice3(h_relu2)   
        if layers_num == 3:
            return [h_relu1, h_relu2, h_relu3]     
        h_relu4 = self.slice4(h_relu3)        
        if layers_num == 4:
            return [h_relu1, h_relu2, h_relu3, h_relu4]     
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out



def define_P(segment_classes, input_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]):
    netP = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'unet_128':
        netP = UnetGenerator(segment_classes, input_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netP = UnetGenerator(segment_classes, input_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    print(netP)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netP.cuda(gpu_ids[0])
    netP.apply(weights_init)
    return netP
    # return init_net(netG, init_type, init_gain, gpu_ids)

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):

    def __init__(self, segment_classes, input_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        output_nc = segment_classes
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        #maybe do some check here with softmax
        self.model = unet_block

    def forward(self, input):
        softmax = torch.nn.Softmax(dim = 1)
        return softmax(self.model(input))

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        
        global printlayer_index
        
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1,output_padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            # printlayer = [PrintLayer(name = str(printlayer_index))]
            # printlayer_index += 1
            # model = printlayer + down + [submodule] + up
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias,output_padding=1)
            # printlayer = [PrintLayer(str(printlayer_index))]
            # printlayer_index += 1
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            # model = printlayer + down + up
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias,output_padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            # printlayer = [PrintLayer(str(printlayer_index))]
            # printlayer_index += 1
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
                # model = printlayer + down + [submodule] + printlayer + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule]  + up
                # model = printlayer + down + [submodule] + printlayer + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        model_output = self.model(x)
        wb,hb = model_output.size()[3],model_output.size()[2]
        wa,ha = x.size()[3],x.size()[2]
        l = int((wb-wa)/2)
        t = int((hb-ha)/2)
        model_output = model_output[:,:,t:t+ha,l:l+wa]
        if self.outermost:
            return model_output
        else:
            return torch.cat([x, model_output], 1)           #if not the outermost block, we concate x and self.model(x) during forward to implement unet
