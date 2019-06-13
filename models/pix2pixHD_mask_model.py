# this is model for fix netP and mask only left eye, right eye, nose and skin
# original loss

import random
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torchvision.transforms as transforms
from PIL import Image

save_tensor_count = 0
save_tensor_list =[]

def softmax2label(featuremap):
    _, label = torch.max(featuremap, dim=1)
    size = label.size()
    label=label.resize_(size[0],1,size[1],size[2])
    return label

def save_tensor(sample, opt):
    global save_tensor_count
    save_tensor_count = save_tensor_count + 1
    save_tensor_list.append(sample.type(torch.FloatTensor))
    print("now the save_tensor_count is " + str(save_tensor_count))
    if save_tensor_count == 330:
        outtensorfile = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), "embed_vector")
        torch.save(save_tensor_list,outtensorfile)
        print("finally, we have saved this tensor ")

class Pix2PixHD_mask_Model(BaseModel):
    def name(self):
        return 'Pix2PixHD_mask_Model'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_l2_loss):
        flags = (True,True,True, use_gan_feat_loss, use_vgg_loss, True, True, use_l2_loss,True,True,True,True)
        def loss_filter(kl_loss,l2_mask_image,g_gan, g_gan_feat, g_vgg, d_real, d_fake, l2_image, loss_parsing,g2_gan,d2_real,d2_fake):
            return [l for (l,f) in zip((kl_loss,l2_mask_image,g_gan,g_gan_feat,g_vgg,d_real,d_fake,l2_image,loss_parsing,g2_gan,d2_real,d2_fake),flags) if f]
        
        return loss_filter
    
    def initialize(self, opt):
        assert opt.vae_encoder == True

        self.name = 'Pix2PixHD_mask_Model'
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        
        if not opt.no_instance:
            netG_input_nc += 1
        
        self.netG = networks.define_embed_bg_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                       opt.n_downsample_global, 9, opt.n_local_enhancers, 
                                       opt.n_blocks_local, opt.norm, 256*5, gpu_ids=self.gpu_ids)
        
        # self.netG = networks.define_embed_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
        #                                opt.n_downsample_global, 9, opt.n_local_enhancers, 
        #                                opt.n_blocks_local, opt.norm, 256*5, gpu_ids=self.gpu_ids)

        # self.netG = networks.define_embed_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
        #                                opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
        #                                opt.n_blocks_local, opt.norm, opt.embed_nc, gpu_ids=self.gpu_ids)



        self.netP = networks.define_P(opt.label_nc, opt.output_nc, 64, "unet_128", opt.norm, use_dropout=True, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            self.netD2 = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, True, gpu_ids=self.gpu_ids)

        embed_feature_size = opt.longSize//2**opt.n_downsample_global 

        self.net_encoder_skin = networks.define_encoder_mask(longsize=256, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_encoder_hair = networks.define_encoder_mask(longsize=256, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_encoder_left_eye = networks.define_encoder_mask(longsize=32, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_encoder_right_eye = networks.define_encoder_mask(longsize=32, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_encoder_mouth = networks.define_encoder_mask(longsize=80, norm=opt.norm, gpu_ids=self.gpu_ids)

        self.net_decoder_skin = networks.define_decoder_mask(longsize=256, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_hair = networks.define_decoder_mask(longsize=256, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_left_eye = networks.define_decoder_mask(longsize=32, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_right_eye = networks.define_decoder_mask(longsize=32, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_mouth = networks.define_decoder_mask(longsize=80, norm=opt.norm, gpu_ids=self.gpu_ids)

        self.net_decoder_skin_image = networks.define_decoder_mask_image(longsize=256, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_hair_image = networks.define_decoder_mask_image(longsize=256, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_left_eye_image = networks.define_decoder_mask_image(longsize=32, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_right_eye_image = networks.define_decoder_mask_image(longsize=32, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.net_decoder_mouth_image = networks.define_decoder_mask_image(longsize=80, norm=opt.norm, gpu_ids=self.gpu_ids)

        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # use for test
        weight_list = [0.5,1,3,3,3,3,10,5,5,5,0.8]
        self.criterionCrossEntropy = torch.nn.CrossEntropyLoss(weight = torch.cuda.FloatTensor(weight_list))
        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            self.load_network(self.net_encoder_skin, 'encoder_skin', opt.which_epoch, pretrained_path)            
            self.load_network(self.net_encoder_hair, 'encoder_hair', opt.which_epoch, pretrained_path)            
            self.load_network(self.net_encoder_left_eye, 'encoder_left_eye', opt.which_epoch, pretrained_path)            
            self.load_network(self.net_encoder_right_eye, 'encoder_right_eye', opt.which_epoch, pretrained_path)          
            self.load_network(self.net_encoder_mouth, 'encoder_mouth', opt.which_epoch, pretrained_path)            

            self.load_network(self.net_decoder_skin, 'decoder_skin', opt.which_epoch, pretrained_path)            
            self.load_network(self.net_decoder_hair, 'decoder_hair', opt.which_epoch, pretrained_path)            
            self.load_network(self.net_decoder_left_eye, 'decoder_left_eye', opt.which_epoch, pretrained_path)            
            self.load_network(self.net_decoder_right_eye, 'decoder_right_eye', opt.which_epoch, pretrained_path)          
            self.load_network(self.net_decoder_mouth, 'decoder_mouth', opt.which_epoch, pretrained_path)            

            self.load_network(self.net_decoder_skin_image, 'decoder_skin_image', opt.which_epoch, pretrained_path)            
            self.load_network(self.net_decoder_hair_image, 'decoder_hair_image', opt.which_epoch, pretrained_path)            
            self.load_network(self.net_decoder_left_eye_image, 'decoder_left_eye_image', opt.which_epoch, pretrained_path)            
            self.load_network(self.net_decoder_right_eye_image, 'decoder_right_eye_image', opt.which_epoch, pretrained_path)
            self.load_network(self.net_decoder_mouth_image, 'decoder_mouth_image', opt.which_epoch, pretrained_path)
            self.load_network(self.netP, 'P', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
                self.load_network(self.netD2, 'D2', opt.which_epoch, pretrained_path)  
        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter( not opt.no_ganFeat_loss, not opt.no_vgg_loss, not opt.no_l2_loss)
            
            # self.criterionKL = torch.nn.KLDivLoss()
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMFM = networks.MFMLoss(self.gpu_ids)
            # weight_list = [0.2,1,2.5,2.5,5,5,1.5,5,5,5,0.2]
            weight_list = [0.2,1,5,5,5,5,3,8,8,8,1]
            self.criterionCrossEntropy = torch.nn.CrossEntropyLoss(weight = torch.cuda.FloatTensor(weight_list))
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids, weights=None)

            self.criterionGM = networks.GramMatrixLoss(self.gpu_ids)
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('KL_embed','L2_mask_image','G_GAN','G_GAN_Feat','G_VGG','D_real','D_fake','L2_image','ParsingLoss','G2_GAN','D2_real','D2_fake')

            # optimizer netP
            # params = list(self.netP.parameters())
            # self.optimizer_netP = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            params_decoder = list(self.net_decoder_skin.parameters()) + list(self.net_decoder_hair.parameters()) + list(self.net_decoder_left_eye.parameters()) + list(self.net_decoder_right_eye.parameters()) + list(self.net_decoder_mouth.parameters())
            params_image_decoder = list(self.net_decoder_skin_image.parameters()) + list(self.net_decoder_hair_image.parameters()) + list(self.net_decoder_left_eye_image.parameters()) + list(self.net_decoder_right_eye_image.parameters()) + list(self.net_decoder_mouth_image.parameters())
            params_encoder = list(self.net_encoder_skin.parameters()) + list(self.net_encoder_hair.parameters()) + list(self.net_encoder_left_eye.parameters()) + list(self.net_encoder_right_eye.parameters()) + list(self.net_encoder_mouth.parameters())

            # self.optimizer_mask_autoencoder = torch.optim.Adam(params_encoder + params_image_decoder, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer sample net
            # params = list(self.netG.parameters()) + params_encoder + params_decoder
            # self.optimizer_sample_net = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer vae net
            # params = list(self.netG.parameters()) + params_decoder + params_encoder
            # self.optimizer_vae_net = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer together of self.optimizer_mask_autoencoder and self.optimizer_vae_net
            params_together = list(self.netG.parameters()) + params_decoder + params_encoder + params_image_decoder
            self.optimizer_G_together = torch.optim.Adam(params_together, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D2
            params = list(self.netD2.parameters())    
            self.optimizer_D2 = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))


    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, image_affine=None, infer=False):             
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1) 
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # affine real images for training
        if image_affine is not None:
            image_affine = Variable(image_affine.data.cuda())

        return input_label, inst_map, real_image, feat_map, image_affine

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def discriminate2(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD2.forward(fake_query)
        else:
            return self.netD2.forward(input_concat)

    def forward(self, bg_image, label, inst, image, feat, image_affine, mask_list, ori_label, infer=False, type="sample_net"):
        if type == "sample_net":
            return self.forward_sample_net(label,inst,image,feat,image_affine,mask_list,infer)
        elif type == "vae_net":
            return self.forward_vae_net(bg_image,label,inst,image,feat,image_affine,mask_list,ori_label,infer)
        else:
            print("false model type ")

    def forward_sample_net(self, label, inst, image, feat, image_affine, infer=False):
        input_label, inst_map, real_image, feat_map, _ = self.encode_input(label, inst, image, feat)
        # start -------------------------------------
        # sample_vector = Variable(torch.randn(len(real_image), self.opt.embed_length).cuda(), requires_grad=False)    

        mask_list = ["1","4","5","6"]
        encode_label_feature = self.netG.forward(input_label,type="label_encoder")
        
        sample_vector1 = Variable(torch.randn(len(input_label),128,4,4).cuda(), requires_grad=False)
        decode_feature1 = self.net_decoder_mask1(sample_vector1)
        encode_label_feature = encode_label_feature + decode_feature1

        sample_vector4 = Variable(torch.randn(len(input_label),1024).cuda(), requires_grad=False)
        decode_feature4 = self.net_decoder_mask4(sample_vector4)
        encode_label_feature[:,:,23:35,18:31] = encode_label_feature[:,:,23:35,18:31] + decode_feature4

        sample_vector5 = Variable(torch.randn(len(input_label),1024).cuda(), requires_grad=False)
        decode_feature5 = self.net_decoder_mask5(sample_vector5)
        encode_label_feature[:,:,23:35,33:46] = encode_label_feature[:,:,23:35,33:46] + decode_feature5

        sample_vector6 = Variable(torch.randn(len(input_label),1024).cuda(), requires_grad=False)
        decode_feature6 = self.net_decoder_mask6(sample_vector6)
        encode_label_feature[:,:,24:44,23:41] = encode_label_feature[:,:,24:44,23:41] + decode_feature6


        fake_image = self.netG.forward(encode_label_feature,type="image_G")


        mask4_image = fake_image[:,:,92:140,72:124]        
        mask5_image = fake_image[:,:,92:140,132:184]
        mask6_image = fake_image[:,:,96:176,92:164]
        mask1 = (label==1).type(torch.cuda.FloatTensor)
        mask1_image = mask1 * fake_image


        reconstruct_mean1, reconstruct_log_var1 = self.net_encoder_mask1(mask1_image)
        loss_l1_vector1 = self.criterionL1(reconstruct_mean1,sample_vector1) * 3
        reconstruct_mean4, reconstruct_log_var4 = self.net_encoder_mask4(mask4_image)
        loss_l1_vector4 = self.criterionL1(reconstruct_mean4,sample_vector4) * 10
        reconstruct_mean5, reconstruct_log_var5 = self.net_encoder_mask5(mask5_image)
        loss_l1_vector5 = self.criterionL1(reconstruct_mean5,sample_vector5) * 10
        reconstruct_mean6, reconstruct_log_var6 = self.net_encoder_mask6(mask6_image)
        loss_l1_vector6 = self.criterionL1(reconstruct_mean6,sample_vector6) * 3
        loss_l1_vector = loss_l1_vector1 + loss_l1_vector4 + loss_l1_vector5 + loss_l1_vector6

        reconstruct_label_feature = self.netP(fake_image)
        reconstruct_label = softmax2label(reconstruct_label_feature)

        # Fake Detection and Loss
        # pred_fake_pool = self.discriminate2(input_label, fake_image, use_pool=True)
        pred_fake_pool = self.netD2.forward(torch.cat((input_label, fake_image.detach()), dim=1))
        loss_D2_fake = self.criterionGAN(pred_fake_pool, False)
        # Real Detection and Loss
        # pred_real = self.discriminate2(input_label, real_image)
        pred_real = self.netD2.forward(torch.cat((input_label, real_image.detach()), dim=1))
        loss_D2_real = self.criterionGAN(pred_real, True)
        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD2.forward(torch.cat((input_label, fake_image), dim=1))        
        loss_G_sample_GAN = self.criterionGAN(pred_fake, True)

        loss_MFM = self.criterionMFM(pred_fake,pred_real) * self.opt.lambda_feat
        
        # gt_label = torch.squeeze(input_label.type(torch.cuda.LongTensor),1)
        # label = label.type(torch.cuda.LongTensor)
        gt_label = torch.squeeze(label.type(torch.cuda.LongTensor),1)
        loss_l1_label = self.criterionCrossEntropy(reconstruct_label_feature,gt_label) * self.opt.lambda_feat


        loss_MFM = loss_MFM.reshape(1)
        loss_D2_fake = loss_D2_fake.reshape(1)
        loss_D2_real = loss_D2_real.reshape(1)
        loss_G_sample_GAN = loss_G_sample_GAN.reshape(1)
        loss_l1_label = loss_l1_label.reshape(1)
        loss_l1_vector = loss_l1_vector.reshape(1)
        zero_tensor=loss_l1_label.clone()
        zero_tensor[0][0]=0

        return self.loss_filter( zero_tensor,zero_tensor,loss_G_sample_GAN,loss_D2_fake,loss_D2_real,zero_tensor,zero_tensor,zero_tensor,zero_tensor,zero_tensor,zero_tensor,loss_l1_label, loss_l1_vector,zero_tensor,loss_MFM ),None if not infer else fake_image, None if not infer else reconstruct_label

    def forward_vae_net(self, bg_image, label, inst, image, feat, image_affine, mask_list, ori_label, infer=False):
        input_label, inst_map, real_image, feat_map, real_bg_image = self.encode_input(label, inst, bg_image, feat, bg_image)  
        # start

        mask4_image = torch.zeros(label.size()[0],3,32,48).cuda()
        mask5_image = torch.zeros(label.size()[0],3,32,48).cuda()
        mask_mouth_image = torch.zeros(label.size()[0],3,80,144).cuda()
        mask_mouth = torch.zeros(label.size()[0],3,80,144).cuda()


        mask_skin = ((label==1)+(label==2)+(label==3)+(label==6)).type(torch.cuda.FloatTensor)
        mask_skin_image = mask_skin * real_image

        mask_hair = (label==10).type(torch.cuda.FloatTensor)
        mask_hair_image = mask_hair * real_image

        mask_mouth_whole = ((label==7)+(label==8)+(label==9)).type(torch.cuda.FloatTensor)

        for batch_index in range(0,label.size()[0]):
            mask4_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][0])-16:int(mask_list[batch_index][0])+16,int(mask_list[batch_index][1])-24:int(mask_list[batch_index][1])+24]
            mask5_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][2])-16:int(mask_list[batch_index][2])+16,int(mask_list[batch_index][3])-24:int(mask_list[batch_index][3])+24]
            mask_mouth_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][4])-40:int(mask_list[batch_index][4])+40,int(mask_list[batch_index][5])-72:int(mask_list[batch_index][5])+72]
            
            mask_mouth[batch_index] = mask_mouth_whole[batch_index,:,int(mask_list[batch_index][4])-40:int(mask_list[batch_index][4])+40,int(mask_list[batch_index][5])-72:int(mask_list[batch_index][5])+72]

        # use masked mouth region
        mask_mouth_image = mask_mouth * mask_mouth_image


        encode_label_feature = self.netG.forward(input_label,type="label_encoder")
        bg_feature = self.netG.forward(real_bg_image,type="bg_encoder")
        mask_bg = (label==0).type(torch.cuda.FloatTensor)
        mask_bg_feature = mask_bg * bg_feature


        loss_mask_image = 0
        loss_KL = 0

        mus4, log_variances4 = self.net_encoder_left_eye(mask4_image)
        variances4 = torch.exp(log_variances4 * 0.5)
        random_sample4 = Variable(torch.randn(mus4.size()).cuda(), requires_grad=True)
        correct_sample4 = random_sample4 * variances4 + mus4
        loss_KL4 = -0.5*torch.sum(-log_variances4.exp() - torch.pow(mus4,2) + log_variances4 + 1)
        reconstruce_mask4_image = self.net_decoder_left_eye_image(correct_sample4)
        loss_mask_image += self.criterionL2(reconstruce_mask4_image, mask4_image.detach()) * 10 
        loss_KL += loss_KL4
        decode_embed_feature4 = self.net_decoder_left_eye(correct_sample4)


        mus5, log_variances5 = self.net_encoder_right_eye(mask5_image)
        variances5 = torch.exp(log_variances5 * 0.5)
        random_sample5 = Variable(torch.randn(mus5.size()).cuda(), requires_grad=True)
        correct_sample5 = random_sample5 * variances5 + mus5
        loss_KL5 = -0.5*torch.sum(-log_variances5.exp() - torch.pow(mus5,2) + log_variances5 + 1)
        reconstruce_mask5_image = self.net_decoder_right_eye_image(correct_sample5)
        loss_mask_image += self.criterionL2(reconstruce_mask5_image, mask5_image.detach()) * 10 
        loss_KL += loss_KL5
        decode_embed_feature5 = self.net_decoder_right_eye(correct_sample5)

        mus_skin, log_variances_skin = self.net_encoder_skin(mask_skin_image)
        variances_skin = torch.exp(log_variances_skin * 0.5)
        random_sample_skin = Variable(torch.randn(mus_skin.size()).cuda(), requires_grad=True)
        correct_sample_skin = random_sample_skin * variances_skin + mus_skin
        loss_KL_skin = -0.5*torch.sum(-log_variances_skin.exp() - torch.pow(mus_skin,2) + log_variances_skin + 1)
        reconstruce_mask_skin_image = self.net_decoder_skin_image(correct_sample_skin)
        reconstruce_mask_skin_image = mask_skin * reconstruce_mask_skin_image
        loss_mask_image += self.criterionL2(reconstruce_mask_skin_image, mask_skin_image.detach()) * 10 
        loss_KL += loss_KL_skin
        decode_embed_feature_skin = self.net_decoder_skin(correct_sample_skin)

        mus_hair, log_variances_hair = self.net_encoder_hair(mask_hair_image)
        variances_hair = torch.exp(log_variances_hair * 0.5)
        random_sample_hair = Variable(torch.randn(mus_hair.size()).cuda(), requires_grad=True)
        correct_sample_hair = random_sample_hair * variances_hair + mus_hair
        loss_KL_hair = -0.5*torch.sum(-log_variances_hair.exp() - torch.pow(mus_hair,2) + log_variances_hair + 1)
        reconstruce_mask_hair_image = self.net_decoder_hair_image(correct_sample_hair)
        reconstruce_mask_hair_image = mask_hair * reconstruce_mask_hair_image
        loss_mask_image += self.criterionL2(reconstruce_mask_hair_image, mask_hair_image.detach()) * 10 
        loss_KL += loss_KL_hair
        decode_embed_feature_hair = self.net_decoder_hair(correct_sample_hair)

        mus_mouth, log_variances_mouth = self.net_encoder_mouth(mask_mouth_image)
        variances_mouth = torch.exp(log_variances_mouth * 0.5)
        random_sample_mouth = Variable(torch.randn(mus_mouth.size()).cuda(), requires_grad=True)
        correct_sample_mouth = random_sample_mouth * variances_mouth + mus_mouth
        loss_KL_mouth = -0.5*torch.sum(-log_variances_mouth.exp() - torch.pow(mus_mouth,2) + log_variances_mouth + 1)
        reconstruce_mask_mouth_image = self.net_decoder_mouth_image(correct_sample_mouth)
        reconstruce_mask_mouth_image = mask_mouth * reconstruce_mask_mouth_image
        # loss_mask_image += self.criterionL2(reconstruce_mask_mouth_image, mask_mouth_image.detach()) * 10
        # mask_mouth_image = mask_mouth * mask_mouth_image 
        loss_mask_image += self.criterionL2(reconstruce_mask_mouth_image, mask_mouth_image.detach()) * 10 
        loss_KL += loss_KL_mouth
        decode_embed_feature_mouth = self.net_decoder_mouth(correct_sample_mouth)



        left_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
        right_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
        mouth_tensor = torch.zeros(encode_label_feature.size()).cuda()



#######################################################################################################
# insert unpaired part
        reorder_left_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
        reorder_right_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
        reorder_mouth_tensor = torch.zeros(encode_label_feature.size()).cuda()


        new_order = torch.randperm(label.size()[0])

        reorder_decode_embed_feature4 = decode_embed_feature4[new_order]
        reorder_decode_embed_feature5 = decode_embed_feature5[new_order]
        reorder_decode_embed_feature_mouth = decode_embed_feature_mouth[new_order]
        reorder_decode_embed_feature_skin = decode_embed_feature_skin[new_order]
        reorder_decode_embed_feature_hair = decode_embed_feature_hair[new_order]

        for batch_index in range(0,label.size()[0]):
            try:
                reorder_left_eye_tensor[batch_index,:,int(mask_list[batch_index][0]/4+0.5)-4:int(mask_list[batch_index][0]/4+0.5)+4,int(mask_list[batch_index][1]/4+0.5)-6:int(mask_list[batch_index][1]/4+0.5)+6] += reorder_decode_embed_feature4[batch_index]
                reorder_right_eye_tensor[batch_index,:,int(mask_list[batch_index][2]/4+0.5)-4:int(mask_list[batch_index][2]/4+0.5)+4,int(mask_list[batch_index][3]/4+0.5)-6:int(mask_list[batch_index][3]/4+0.5)+6] += reorder_decode_embed_feature5[batch_index]
                reorder_mouth_tensor[batch_index,:,int(mask_list[batch_index][4]/4+0.5)-10:int(mask_list[batch_index][4]/4+0.5)+10,int(mask_list[batch_index][5]/4+0.5)-18:int(mask_list[batch_index][5]/4+0.5)+18] += reorder_decode_embed_feature_mouth[batch_index]
            except:
                print("wrong0 ! ")

        reconstruct_transfer_face = self.netG.forward(torch.cat((encode_label_feature,reorder_left_eye_tensor,reorder_right_eye_tensor,reorder_decode_embed_feature_skin,reorder_decode_embed_feature_hair,reorder_mouth_tensor),1),type="image_G")

        reconstruct_transfer_image = self.netG.forward(torch.cat((reconstruct_transfer_face,mask_bg_feature),1),type="bg_decoder")


        parsing_label_feature = self.netP(reconstruct_transfer_image)
        parsing_label = softmax2label(parsing_label_feature)
        gt_label = torch.squeeze(ori_label.type(torch.cuda.LongTensor),1)
        loss_parsing = self.criterionCrossEntropy(parsing_label_feature,gt_label)*self.opt.lambda_feat


        pred_fake2_pool = self.netD2.forward(torch.cat((input_label, reconstruct_transfer_image.detach()), dim=1))
        loss_D2_fake = self.criterionGAN(pred_fake2_pool, False)
        # Real Detection and Loss
        # pred_real = self.discriminate(input_label, real_image)
        pred_real2 = self.netD2.forward(torch.cat((input_label, real_image.detach()), dim=1))
        loss_D2_real = self.criterionGAN(pred_real2, True)
        # GAN loss (Fake Passability Loss)        
        pred_fake2 = self.netD2.forward(torch.cat((input_label, reconstruct_transfer_image), dim=1))        
        loss_G2_GAN = self.criterionGAN(pred_fake2, True)





########################################################################################################


        for batch_index in range(0,label.size()[0]):
            try:
                left_eye_tensor[batch_index,:,int(mask_list[batch_index][0]/4+0.5)-4:int(mask_list[batch_index][0]/4+0.5)+4,int(mask_list[batch_index][1]/4+0.5)-6:int(mask_list[batch_index][1]/4+0.5)+6] += decode_embed_feature4[batch_index]
                right_eye_tensor[batch_index,:,int(mask_list[batch_index][2]/4+0.5)-4:int(mask_list[batch_index][2]/4+0.5)+4,int(mask_list[batch_index][3]/4+0.5)-6:int(mask_list[batch_index][3]/4+0.5)+6] += decode_embed_feature5[batch_index]
                mouth_tensor[batch_index,:,int(mask_list[batch_index][4]/4+0.5)-10:int(mask_list[batch_index][4]/4+0.5)+10,int(mask_list[batch_index][5]/4+0.5)-18:int(mask_list[batch_index][5]/4+0.5)+18] += decode_embed_feature_mouth[batch_index]
            except:
                print("wrong ! ")

        # loss_KL4 = -0.5*torch.sum(-log_variances4.exp() - torch.pow(mus4,2) + log_variances4 + 1, 1)

        reconstruct_face = self.netG.forward(torch.cat((encode_label_feature,left_eye_tensor,right_eye_tensor,decode_embed_feature_skin,decode_embed_feature_hair,mouth_tensor),1),type="image_G")

        reconstruct_image = self.netG.forward(torch.cat((reconstruct_face,mask_bg_feature),1),type="bg_decoder")        


        # reconstruce_part image

        mask_left_eye = (label==4).type(torch.cuda.FloatTensor)
        mask_right_eye = (label==5).type(torch.cuda.FloatTensor)
        mask_mouth = ((label==7)+(label==8)+(label==9)).type(torch.cuda.FloatTensor)

        loss_L2_image = 0
        for batch_index in range(0,label.size()[0]):
            loss_L2_image += self.criterionL2( mask_left_eye*reconstruct_image, mask_left_eye*real_image) * 10 
            loss_L2_image += self.criterionL2( mask_right_eye*reconstruct_image, mask_right_eye*real_image) * 10 
            loss_L2_image += self.criterionL2( mask_skin*reconstruct_image, mask_skin*real_image) * 5 
            loss_L2_image += self.criterionL2( mask_hair*reconstruct_image, mask_hair*real_image) * 5
            loss_L2_image += self.criterionL2( mask_mouth*reconstruct_image, mask_mouth*real_image) * 10 
            loss_L2_image += self.criterionL2( reconstruct_image, real_bg_image ) * 10

        # Fake Detection and Loss
        # pred_fake_pool = self.discriminate(input_label, reconstruct_image, use_pool=True)
        pred_fake_pool = self.netD.forward(torch.cat((input_label, reconstruct_image.detach()), dim=1))
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        # Real Detection and Loss
        # pred_real = self.discriminate(input_label, real_image)
        pred_real = self.netD.forward(torch.cat((input_label, real_image.detach()), dim=1))
        loss_D_real = self.criterionGAN(pred_real, True)
        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label, reconstruct_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        

        # mask = (label==1) + (label==4) + (label==5) + (label==6)
        # mask = mask.type(torch.cuda.FloatTensor)
        # mask_reconstruct_image = mask * reconstruct_image
        # mask_real_image = mask * real_image
        # loss_L2_image = self.criterionL2(mask_reconstruct_image,mask_real_image)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        
        all_mask_tensor = torch.cat((mask_left_eye,mask_right_eye,mask_skin,mask_hair,mask_mouth),1)
        # mask_weight_tensor = torch.zeros(5)
        # mask_weight_tensor[0] = 10
        # mask_weight_tensor[1] = 10
        # mask_weight_tensor[2] = 5
        # mask_weight_tensor[3] = 1
        # mask_weight_tensor[4] = 10
        mask_weight_list = [10,10,5,5,10]
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG += self.criterionVGG(reconstruct_image, real_image, all_mask_tensor, mask_weights = mask_weight_list) * self.opt.lambda_feat * 3
            # loss_G_VGG += self.criterionVGG(reconstruct_image, real_image, mask4, weights = [1.0/4,1.0/4,1.0/4,1.0/8,1.0/8]) * self.opt.lambda_feat * 10
            
        return self.loss_filter( loss_KL,loss_mask_image,loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_L2_image, loss_parsing, loss_G2_GAN, loss_D2_real, loss_D2_fake), None if not infer else reconstruct_image, None if not infer else reconstruce_mask4_image, None if not infer else reconstruce_mask5_image, None if not infer else reconstruce_mask_skin_image, None if not infer else reconstruce_mask_hair_image, None if not infer else reconstruce_mask_mouth_image, None if not infer else reconstruct_transfer_image, None if not infer else parsing_label
        
    def inference(self, bg_contentimage, label2, mask2_list, image, label, mask_list):

        # same z for different mask 

        # Encode Inputs        

        input_label, inst_map, real_image, _, affine_real_image = self.encode_input(Variable(label), None, Variable(image), infer=True)

        input_label2, _, real_content_image, _, _ = self.encode_input(Variable(label2), None, Variable(bg_contentimage), infer=True)

        assert torch.__version__.startswith('0.4')
        with torch.no_grad():
            # first go through the encoder and decoder
            self.net_encoder_left_eye.eval()
            self.net_encoder_right_eye.eval()
            self.net_encoder_skin.eval()
            self.net_encoder_hair.eval()
            self.net_encoder_mouth.eval()
            self.net_decoder_left_eye.eval()
            self.net_decoder_right_eye.eval()
            self.net_decoder_skin.eval()
            self.net_decoder_hair.eval()
            self.net_decoder_mouth.eval()
            self.netG.eval()

            mask4_image = torch.zeros(label.size()[0],3,32,48).cuda()
            mask5_image = torch.zeros(label.size()[0],3,32,48).cuda()
            mask_mouth_image = torch.zeros(label.size()[0],3,80,144).cuda()
            mask_mouth = torch.zeros(label.size()[0],3,80,144).cuda()            

            mask_skin = ((label==1)+(label==2)+(label==3)+(label==6)).type(torch.cuda.FloatTensor)
            mask_skin_image = mask_skin * real_image
            mask_hair = (label==10).type(torch.cuda.FloatTensor)
            mask_hair_image = mask_hair * real_image

            mask_mouth_whole = ((label==7)+(label==8)+(label==9)).type(torch.cuda.FloatTensor)


            bg_content_feature = self.netG.forward(real_content_image,type="bg_encoder")
            mask_content_bg = (label2==0).type(torch.cuda.FloatTensor)
            mask_content_bg_feature = mask_content_bg * bg_content_feature

            bg_style_feature = self.netG.forward(real_image,type="bg_encoder")
            mask_style_bg = (label==0).type(torch.cuda.FloatTensor)
            mask_style_bg_feature = mask_style_bg * bg_style_feature



            for batch_index in range(0,label.size()[0]):
                mask4_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][0])-16:int(mask_list[batch_index][0])+16,int(mask_list[batch_index][1])-24:int(mask_list[batch_index][1])+24]
                mask5_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][2])-16:int(mask_list[batch_index][2])+16,int(mask_list[batch_index][3])-24:int(mask_list[batch_index][3])+24]
                mask_mouth_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][4])-40:int(mask_list[batch_index][4])+40,int(mask_list[batch_index][5])-72:int(mask_list[batch_index][5])+72]

                mask_mouth[batch_index] = mask_mouth_whole[batch_index,:,int(mask_list[batch_index][4])-40:int(mask_list[batch_index][4])+40,int(mask_list[batch_index][5])-72:int(mask_list[batch_index][5])+72]

            mask_mouth_image = mask_mouth * mask_mouth_image


            encode_label_feature = self.netG.forward(input_label,type="label_encoder")
            encode_label2_feature = self.netG.forward(input_label2,type="label_encoder")

            if self.opt.random_embed == False:
                # means we use input image to generate embed vector
                with torch.no_grad():
                    correct_sample4, log_variances4 = self.net_encoder_left_eye(mask4_image)
                    decode_embed_feature4 = self.net_decoder_left_eye(correct_sample4)

                    correct_sample5, log_variances5 = self.net_encoder_right_eye(mask5_image)
                    decode_embed_feature5 = self.net_decoder_right_eye(correct_sample5)

                    correct_sample_skin, log_variances_skin = self.net_encoder_skin(mask_skin_image)
                    decode_embed_feature_skin = self.net_decoder_skin(correct_sample_skin)

                    correct_sample_hair, log_variances_hair = self.net_encoder_hair(mask_hair_image)
                    decode_embed_feature_hair = self.net_decoder_hair(correct_sample_hair)

                    correct_sample_mouth, log_variances_mouth = self.net_encoder_mouth(mask_mouth_image)
                    decode_embed_feature_mouth = self.net_decoder_mouth(correct_sample_mouth)

            else:
                with torch.no_grad():
                    mus4, log_variances4 = self.net_encoder_left_eye(mask4_image)
                    correct_sample4 = Variable(torch.randn(mus4.size()).cuda(), requires_grad=True)
                    decode_embed_feature4 = self.net_decoder_left_eye(correct_sample4)

                    mus5, log_variances5 = self.net_encoder_right_eye(mask5_image)
                    correct_sample5 = Variable(torch.randn(mus5.size()).cuda(), requires_grad=True)
                    decode_embed_feature5 = self.net_decoder_right_eye(correct_sample5)

                    mus_skin, log_variances_skin = self.net_encoder_skin(mask_skin_image)
                    correct_sample_skin = Variable(torch.randn(mus_skin.size()).cuda(), requires_grad=True)
                    decode_embed_feature_skin = self.net_decoder_skin(correct_sample_skin)

                    mus_hair, log_variances_hair = self.net_encoder_hair(mask_hair_image)
                    correct_sample_hair = Variable(torch.randn(mus_hair.size()).cuda(), requires_grad=True)
                    decode_embed_feature_hair = self.net_decoder_hair(correct_sample_hair)

                    mus_mouth, log_variances_mouth = self.net_encoder_mouth(mask_mouth_image)
                    correct_sample_mouth = Variable(torch.randn(mus_mouth.size()).cuda(), requires_grad=True)
                    decode_embed_feature_mouth = self.net_decoder_mouth(correct_sample_mouth)

            # save_tensor(correct_sample, self.opt)

            left_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
            right_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
            mouth_tensor = torch.zeros(encode_label_feature.size()).cuda()
            for batch_index in range(0,label.size()[0]):
                try:
                    left_eye_tensor[batch_index,:,int(mask_list[batch_index][0]/4+0.5)-4:int(mask_list[batch_index][0]/4+0.5)+4,int(mask_list[batch_index][1]/4+0.5)-6:int(mask_list[batch_index][1]/4+0.5)+6] += decode_embed_feature4[batch_index]
                    right_eye_tensor[batch_index,:,int(mask_list[batch_index][2]/4+0.5)-4:int(mask_list[batch_index][2]/4+0.5)+4,int(mask_list[batch_index][3]/4+0.5)-6:int(mask_list[batch_index][3]/4+0.5)+6] += decode_embed_feature5[batch_index]
                    mouth_tensor[batch_index,:,int(mask_list[batch_index][4]/4+0.5)-10:int(mask_list[batch_index][4]/4+0.5)+10,int(mask_list[batch_index][5]/4+0.5)-18:int(mask_list[batch_index][5]/4+0.5)+18] += decode_embed_feature_mouth[batch_index]
                except:
                    print("wrong ! ")

            fake_style_face = self.netG.forward(torch.cat((encode_label_feature,left_eye_tensor,right_eye_tensor,decode_embed_feature_skin,decode_embed_feature_hair,mouth_tensor),1),type="image_G")

            left_eye_tensor = torch.zeros(encode_label2_feature.size()).cuda()
            right_eye_tensor = torch.zeros(encode_label2_feature.size()).cuda()
            mouth_tensor = torch.zeros(encode_label2_feature.size()).cuda()
            for batch_index in range(0,label.size()[0]):
                try:
                    left_eye_tensor[batch_index,:,int(mask2_list[batch_index][0]/4+0.5)-4:int(mask2_list[batch_index][0]/4+0.5)+4,int(mask2_list[batch_index][1]/4+0.5)-6:int(mask2_list[batch_index][1]/4+0.5)+6] += decode_embed_feature4[batch_index]
                    right_eye_tensor[batch_index,:,int(mask2_list[batch_index][2]/4+0.5)-4:int(mask2_list[batch_index][2]/4+0.5)+4,int(mask2_list[batch_index][3]/4+0.5)-6:int(mask2_list[batch_index][3]/4+0.5)+6] += decode_embed_feature5[batch_index]
                    mouth_tensor[batch_index,:,int(mask2_list[batch_index][4]/4+0.5)-10:int(mask2_list[batch_index][4]/4+0.5)+10,int(mask2_list[batch_index][5]/4+0.5)-18:int(mask2_list[batch_index][5]/4+0.5)+18] += decode_embed_feature_mouth[batch_index]
                except:
                    print("wrong ! ")

            fake_content_face = self.netG.forward(torch.cat((encode_label2_feature,left_eye_tensor,right_eye_tensor,decode_embed_feature_skin,decode_embed_feature_hair,mouth_tensor),1),type="image_G")


            reconstruct_fake_style = self.netG.forward(torch.cat((fake_style_face,mask_style_bg_feature),1),type="bg_decoder")
            reconstruct_fake_content = self.netG.forward(torch.cat((fake_content_face,mask_content_bg_feature),1),type="bg_decoder")

        return reconstruct_fake_style, reconstruct_fake_content, mask_mouth_image

        # return fake_image,fake_image2,mask_mouth_image

    def inference_2image(self, bg_image, label, mask_list, ori_label):

        # same z for different mask 

        # Encode Inputs        

        input_label, inst_map, real_image, _, _ = self.encode_input(Variable(label), None, Variable(bg_image), infer=True)

        assert torch.__version__.startswith('0.4')
        with torch.no_grad():
            # first go through the encoder and decoder
            self.net_encoder_left_eye.eval()
            self.net_encoder_right_eye.eval()
            self.net_encoder_skin.eval()
            self.net_encoder_hair.eval()
            self.net_encoder_mouth.eval()
            self.net_decoder_left_eye.eval()
            self.net_decoder_right_eye.eval()
            self.net_decoder_skin.eval()
            self.net_decoder_hair.eval()
            self.net_decoder_mouth.eval()
            self.netG.eval()

            mask4_image = torch.zeros(label.size()[0],3,32,48).cuda()
            mask5_image = torch.zeros(label.size()[0],3,32,48).cuda()
            mask_mouth_image = torch.zeros(label.size()[0],3,80,144).cuda()
            mask_mouth = torch.zeros(label.size()[0],3,80,144).cuda()            

            mask_skin = ((label==1)+(label==2)+(label==3)+(label==6)).type(torch.cuda.FloatTensor)
            mask_skin_image = mask_skin * real_image
            mask_hair = (label==10).type(torch.cuda.FloatTensor)
            mask_hair_image = mask_hair * real_image

            mask_mouth_whole = ((label==7)+(label==8)+(label==9)).type(torch.cuda.FloatTensor)


            bg_feature = self.netG.forward(real_image,type="bg_encoder")
            mask_bg = (label==0).type(torch.cuda.FloatTensor)
            mask_bg_feature = mask_bg * bg_feature





            for batch_index in range(0,label.size()[0]):
                mask4_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][0])-16:int(mask_list[batch_index][0])+16,int(mask_list[batch_index][1])-24:int(mask_list[batch_index][1])+24]
                mask5_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][2])-16:int(mask_list[batch_index][2])+16,int(mask_list[batch_index][3])-24:int(mask_list[batch_index][3])+24]
                mask_mouth_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][4])-40:int(mask_list[batch_index][4])+40,int(mask_list[batch_index][5])-72:int(mask_list[batch_index][5])+72]

                mask_mouth[batch_index] = mask_mouth_whole[batch_index,:,int(mask_list[batch_index][4])-40:int(mask_list[batch_index][4])+40,int(mask_list[batch_index][5])-72:int(mask_list[batch_index][5])+72]

            mask_mouth_image = mask_mouth * mask_mouth_image


            encode_label_feature = self.netG.forward(input_label,type="label_encoder")

            with torch.no_grad():
                correct_sample4, log_variances4 = self.net_encoder_left_eye(mask4_image)
                decode_embed_feature4 = self.net_decoder_left_eye(correct_sample4)

                correct_sample5, log_variances5 = self.net_encoder_right_eye(mask5_image)
                decode_embed_feature5 = self.net_decoder_right_eye(correct_sample5)

                torch.save(decode_embed_feature4,"a_tesnor.pth")
                torch.save(decode_embed_feature5,"b_tesnor.pth")

                correct_sample_skin, log_variances_skin = self.net_encoder_skin(mask_skin_image)
                decode_embed_feature_skin = self.net_decoder_skin(correct_sample_skin)

                correct_sample_hair, log_variances_hair = self.net_encoder_hair(mask_hair_image)
                decode_embed_feature_hair = self.net_decoder_hair(correct_sample_hair)

                correct_sample_mouth, log_variances_mouth = self.net_encoder_mouth(mask_mouth_image)
                decode_embed_feature_mouth = self.net_decoder_mouth(correct_sample_mouth)




            reorder_left_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
            reorder_right_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
            reorder_mouth_tensor = torch.zeros(encode_label_feature.size()).cuda()


            new_order = torch.randperm(label.size()[0])
            if random.random() > 0:
                new_order[0] = 1
                new_order[1] = 0

            reorder_decode_embed_feature4 = decode_embed_feature4[new_order]
            reorder_decode_embed_feature5 = decode_embed_feature5[new_order]
            reorder_decode_embed_feature_mouth = decode_embed_feature_mouth[new_order]
            reorder_decode_embed_feature_skin = decode_embed_feature_skin[new_order]
            reorder_decode_embed_feature_hair = decode_embed_feature_hair[new_order]

            for batch_index in range(0,label.size()[0]):
                try:
                    reorder_left_eye_tensor[batch_index,:,int(mask_list[batch_index][0]/4+0.5)-4:int(mask_list[batch_index][0]/4+0.5)+4,int(mask_list[batch_index][1]/4+0.5)-6:int(mask_list[batch_index][1]/4+0.5)+6] += reorder_decode_embed_feature4[batch_index]
                    reorder_right_eye_tensor[batch_index,:,int(mask_list[batch_index][2]/4+0.5)-4:int(mask_list[batch_index][2]/4+0.5)+4,int(mask_list[batch_index][3]/4+0.5)-6:int(mask_list[batch_index][3]/4+0.5)+6] += reorder_decode_embed_feature5[batch_index]
                    reorder_mouth_tensor[batch_index,:,int(mask_list[batch_index][4]/4+0.5)-10:int(mask_list[batch_index][4]/4+0.5)+10,int(mask_list[batch_index][5]/4+0.5)-18:int(mask_list[batch_index][5]/4+0.5)+18] += reorder_decode_embed_feature_mouth[batch_index]
                except:
                    print("wrong0 ! ")



            fake_face = self.netG.forward(torch.cat((encode_label_feature,reorder_left_eye_tensor,reorder_right_eye_tensor,reorder_decode_embed_feature_skin,reorder_decode_embed_feature_hair,reorder_mouth_tensor),1),type="image_G")
            fake_image = self.netG.forward(torch.cat((fake_face,mask_bg_feature),1),type="bg_decoder")

        return fake_image



    def inference_multi_embed(self, label, inst, image, mask_list, label2, mask2_list):

        # same z for different mask 

        # Encode Inputs        

        input_label, inst_map, real_image, _, affine_real_image = self.encode_input(Variable(label), Variable(inst), Variable(image), infer=True)

        input_label2, _, _, _, _ = self.encode_input(Variable(label2), Variable(inst), Variable(image), infer=True)

        assert torch.__version__.startswith('0.4')
        with torch.no_grad():
            # first go through the encoder and decoder
            self.net_encoder_left_eye.eval()
            self.net_encoder_right_eye.eval()
            self.net_encoder_skin.eval()
            self.net_encoder_hair.eval()
            self.net_encoder_mouth.eval()
            self.net_decoder_left_eye.eval()
            self.net_decoder_right_eye.eval()
            self.net_decoder_skin.eval()
            self.net_decoder_hair.eval()
            self.net_decoder_mouth.eval()
            self.netG.eval()

            # load 5 part embed images and synthesis 5 mask input_label2[0:4]
            # left_eye, right_eye, skin, hair, mouth 


            mask4_image = torch.zeros(1,3,32,48).cuda()
            mask5_image = torch.zeros(1,3,32,48).cuda()
            mask_mouth_image = torch.zeros(1,3,80,144).cuda()
            mask_skin = ((label[2:3]==1)+(label[2:3]==2)+(label[2:3]==3)+(label[2:3]==6)).type(torch.cuda.FloatTensor)
            mask_skin_image = mask_skin * real_image[2:3]
            mask_hair = (label[3:4]==10).type(torch.cuda.FloatTensor)
            mask_hair_image = mask_hair * real_image[3:4]

            
            mask4_image[0] = real_image[0,:,int(mask_list[0][0])-16:int(mask_list[0][0])+16,int(mask_list[0][1])-24:int(mask_list[0][1])+24]
            mask5_image[0] = real_image[1,:,int(mask_list[1][2])-16:int(mask_list[1][2])+16,int(mask_list[1][3])-24:int(mask_list[1][3])+24]
            mask_mouth_image[0] = real_image[4,:,int(mask_list[4][4])-40:int(mask_list[4][4])+40,int(mask_list[4][5])-72:int(mask_list[4][5])+72]


            mask4_image = mask4_image.expand(5,3,32,48)
            mask5_image = mask5_image.expand(5,3,32,48)
            mask_skin_image = mask_skin_image.expand(5,3,256,256)
            mask_hair_image = mask_hair_image.expand(5,3,256,256)
            mask_mouth_image = mask_mouth_image.expand(5,3,80,144)

            encode_label2_feature = self.netG.forward(input_label2,type="label_encoder")

            assert self.opt.random_embed == False 
            with torch.no_grad():
                correct_sample4, log_variances4 = self.net_encoder_left_eye(mask4_image)
                decode_embed_feature4 = self.net_decoder_left_eye(correct_sample4)

                correct_sample5, log_variances5 = self.net_encoder_right_eye(mask5_image)
                decode_embed_feature5 = self.net_decoder_right_eye(correct_sample5)

                correct_sample_skin, log_variances_skin = self.net_encoder_skin(mask_skin_image)
                decode_embed_feature_skin = self.net_decoder_skin(correct_sample_skin)

                correct_sample_hair, log_variances_hair = self.net_encoder_hair(mask_hair_image)
                decode_embed_feature_hair = self.net_decoder_hair(correct_sample_hair)

                correct_sample_mouth, log_variances_mouth = self.net_encoder_mouth(mask_mouth_image)
                decode_embed_feature_mouth = self.net_decoder_mouth(correct_sample_mouth)



            left_eye_tensor = torch.zeros(encode_label2_feature.size()).cuda()
            right_eye_tensor = torch.zeros(encode_label2_feature.size()).cuda()
            mouth_tensor = torch.zeros(encode_label2_feature.size()).cuda()
            for batch_index in range(0,label.size()[0]):
                try:
                    left_eye_tensor[batch_index,:,int(mask2_list[batch_index][0]/4+0.5)-4:int(mask2_list[batch_index][0]/4+0.5)+4,int(mask2_list[batch_index][1]/4+0.5)-6:int(mask2_list[batch_index][1]/4+0.5)+6] += decode_embed_feature4[batch_index]
                    right_eye_tensor[batch_index,:,int(mask2_list[batch_index][2]/4+0.5)-4:int(mask2_list[batch_index][2]/4+0.5)+4,int(mask2_list[batch_index][3]/4+0.5)-6:int(mask2_list[batch_index][3]/4+0.5)+6] += decode_embed_feature5[batch_index]
                    mouth_tensor[batch_index,:,int(mask2_list[batch_index][4]/4+0.5)-10:int(mask2_list[batch_index][4]/4+0.5)+10,int(mask2_list[batch_index][5]/4+0.5)-18:int(mask2_list[batch_index][5]/4+0.5)+18] += decode_embed_feature_mouth[batch_index]
                except:
                    print("wrong ! ")

            fake_image2 = self.netG.forward(torch.cat((encode_label2_feature,left_eye_tensor,right_eye_tensor,decode_embed_feature_skin,decode_embed_feature_hair,mouth_tensor),1),type="image_G")

        return fake_image2

    def inference_encode(self, path, image, label, mask_list):

        # same z for different mask 

        # Encode Inputs        
        base_path = os.path.join(self.opt.results_dir, self.opt.name, '%s_%s' % (self.opt.phase, self.opt.which_epoch))

        input_label, inst_map, real_image, _, affine_real_image = self.encode_input(Variable(label), None, Variable(image), infer=True)

        assert torch.__version__.startswith('0.4')
        with torch.no_grad():
            # first go through the encoder and decoder
            self.net_encoder_left_eye.eval()
            self.net_encoder_right_eye.eval()
            self.net_encoder_skin.eval()
            self.net_encoder_hair.eval()
            self.net_encoder_mouth.eval()
            self.net_decoder_left_eye.eval()
            self.net_decoder_right_eye.eval()
            self.net_decoder_skin.eval()
            self.net_decoder_hair.eval()
            self.net_decoder_mouth.eval()
            self.netG.eval()

            mask4_image = torch.zeros(label.size()[0],3,32,48).cuda()
            mask5_image = torch.zeros(label.size()[0],3,32,48).cuda()
            mask_mouth_image = torch.zeros(label.size()[0],3,80,144).cuda()
            mask_mouth = torch.zeros(label.size()[0],3,80,144).cuda()            

            mask_skin = ((label==1)+(label==2)+(label==3)+(label==6)).type(torch.cuda.FloatTensor)
            mask_skin_image = mask_skin * real_image
            mask_hair = (label==10).type(torch.cuda.FloatTensor)
            mask_hair_image = mask_hair * real_image

            mask_mouth_whole = ((label==7)+(label==8)+(label==9)).type(torch.cuda.FloatTensor)


            bg_style_feature = self.netG.forward(real_image,type="bg_encoder")
            mask_style_bg = (label==0).type(torch.cuda.FloatTensor)
            mask_style_bg_feature = mask_style_bg * bg_style_feature



            for batch_index in range(0,label.size()[0]):
                mask4_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][0])-16:int(mask_list[batch_index][0])+16,int(mask_list[batch_index][1])-24:int(mask_list[batch_index][1])+24]
                mask5_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][2])-16:int(mask_list[batch_index][2])+16,int(mask_list[batch_index][3])-24:int(mask_list[batch_index][3])+24]
                mask_mouth_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][4])-40:int(mask_list[batch_index][4])+40,int(mask_list[batch_index][5])-72:int(mask_list[batch_index][5])+72]

                mask_mouth[batch_index] = mask_mouth_whole[batch_index,:,int(mask_list[batch_index][4])-40:int(mask_list[batch_index][4])+40,int(mask_list[batch_index][5])-72:int(mask_list[batch_index][5])+72]

            mask_mouth_image = mask_mouth * mask_mouth_image


            encode_label_feature = self.netG.forward(input_label,type="label_encoder")

            with torch.no_grad():
                correct_sample4, log_variances4 = self.net_encoder_left_eye(mask4_image)
                decode_embed_feature4 = self.net_decoder_left_eye(correct_sample4)

                correct_sample5, log_variances5 = self.net_encoder_right_eye(mask5_image)
                decode_embed_feature5 = self.net_decoder_right_eye(correct_sample5)

                correct_sample_skin, log_variances_skin = self.net_encoder_skin(mask_skin_image)
                decode_embed_feature_skin = self.net_decoder_skin(correct_sample_skin)

                correct_sample_hair, log_variances_hair = self.net_encoder_hair(mask_hair_image)
                decode_embed_feature_hair = self.net_decoder_hair(correct_sample_hair)

                correct_sample_mouth, log_variances_mouth = self.net_encoder_mouth(mask_mouth_image)
                decode_embed_feature_mouth = self.net_decoder_mouth(correct_sample_mouth)
                print("path is ")
                image_name = path[0].split('/')[-1].replace('.png','').replace('.jpg','')
                save_path = base_path + "/encode_tensor/" + image_name
                print(save_path)

                if os.path.exists(save_path) == False:
                    os.mkdir(save_path)
                torch.save(decode_embed_feature4,save_path+"/left_eye")
                torch.save(decode_embed_feature5,save_path+"/right_eye")
                torch.save(decode_embed_feature_skin,save_path+"/skin")
                torch.save(decode_embed_feature_hair,save_path+"/hair")
                torch.save(decode_embed_feature_mouth,save_path+"/mouth")

            left_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
            right_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
            mouth_tensor = torch.zeros(encode_label_feature.size()).cuda()
            for batch_index in range(0,label.size()[0]):
                try:
                    left_eye_tensor[batch_index,:,int(mask_list[batch_index][0]/4+0.5)-4:int(mask_list[batch_index][0]/4+0.5)+4,int(mask_list[batch_index][1]/4+0.5)-6:int(mask_list[batch_index][1]/4+0.5)+6] += decode_embed_feature4[batch_index]
                    right_eye_tensor[batch_index,:,int(mask_list[batch_index][2]/4+0.5)-4:int(mask_list[batch_index][2]/4+0.5)+4,int(mask_list[batch_index][3]/4+0.5)-6:int(mask_list[batch_index][3]/4+0.5)+6] += decode_embed_feature5[batch_index]
                    mouth_tensor[batch_index,:,int(mask_list[batch_index][4]/4+0.5)-10:int(mask_list[batch_index][4]/4+0.5)+10,int(mask_list[batch_index][5]/4+0.5)-18:int(mask_list[batch_index][5]/4+0.5)+18] += decode_embed_feature_mouth[batch_index]
                except:
                    print("wrong ! ")

            fake_style_face = self.netG.forward(torch.cat((encode_label_feature,left_eye_tensor,right_eye_tensor,decode_embed_feature_skin,decode_embed_feature_hair,mouth_tensor),1),type="image_G")


            reconstruct_fake_style = self.netG.forward(torch.cat((fake_style_face,mask_style_bg_feature),1),type="bg_decoder")

        return reconstruct_fake_style


    def inference_generate(self, path, image, label, mask_list):

        # same z for different mask 

        # Encode Inputs        
        base_path = os.path.join(self.opt.results_dir, self.opt.name, '%s_%s' % (self.opt.phase, self.opt.which_epoch))

        input_label, inst_map, real_image, _, affine_real_image = self.encode_input(Variable(label), None, Variable(image), infer=True)

        assert torch.__version__.startswith('0.4')
        with torch.no_grad():
            # first go through the encoder and decoder
            self.net_encoder_left_eye.eval()
            self.net_encoder_right_eye.eval()
            self.net_encoder_skin.eval()
            self.net_encoder_hair.eval()
            self.net_encoder_mouth.eval()
            self.net_decoder_left_eye.eval()
            self.net_decoder_right_eye.eval()
            self.net_decoder_skin.eval()
            self.net_decoder_hair.eval()
            self.net_decoder_mouth.eval()
            self.netG.eval()

            mask4_image = torch.zeros(label.size()[0],3,32,48).cuda()
            mask5_image = torch.zeros(label.size()[0],3,32,48).cuda()
            mask_mouth_image = torch.zeros(label.size()[0],3,80,144).cuda()
            mask_mouth = torch.zeros(label.size()[0],3,80,144).cuda()            

            mask_skin = ((label==1)+(label==2)+(label==3)+(label==6)).type(torch.cuda.FloatTensor)
            mask_skin_image = mask_skin * real_image
            mask_hair = (label==10).type(torch.cuda.FloatTensor)
            mask_hair_image = mask_hair * real_image

            mask_mouth_whole = ((label==7)+(label==8)+(label==9)).type(torch.cuda.FloatTensor)


            bg_style_feature = self.netG.forward(real_image,type="bg_encoder")
            mask_style_bg = (label==0).type(torch.cuda.FloatTensor)
            mask_style_bg_feature = mask_style_bg * bg_style_feature



            for batch_index in range(0,label.size()[0]):
                mask4_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][0])-16:int(mask_list[batch_index][0])+16,int(mask_list[batch_index][1])-24:int(mask_list[batch_index][1])+24]
                mask5_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][2])-16:int(mask_list[batch_index][2])+16,int(mask_list[batch_index][3])-24:int(mask_list[batch_index][3])+24]
                mask_mouth_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][4])-40:int(mask_list[batch_index][4])+40,int(mask_list[batch_index][5])-72:int(mask_list[batch_index][5])+72]

                mask_mouth[batch_index] = mask_mouth_whole[batch_index,:,int(mask_list[batch_index][4])-40:int(mask_list[batch_index][4])+40,int(mask_list[batch_index][5])-72:int(mask_list[batch_index][5])+72]

            mask_mouth_image = mask_mouth * mask_mouth_image


            encode_label_feature = self.netG.forward(input_label,type="label_encoder")

            with torch.no_grad():
                correct_sample4, log_variances4 = self.net_encoder_left_eye(mask4_image)
                decode_embed_feature4 = self.net_decoder_left_eye(correct_sample4)

                correct_sample5, log_variances5 = self.net_encoder_right_eye(mask5_image)
                decode_embed_feature5 = self.net_decoder_right_eye(correct_sample5)

                correct_sample_skin, log_variances_skin = self.net_encoder_skin(mask_skin_image)
                decode_embed_feature_skin = self.net_decoder_skin(correct_sample_skin)

                correct_sample_hair, log_variances_hair = self.net_encoder_hair(mask_hair_image)
                decode_embed_feature_hair = self.net_decoder_hair(correct_sample_hair)

                correct_sample_mouth, log_variances_mouth = self.net_encoder_mouth(mask_mouth_image)
                decode_embed_feature_mouth = self.net_decoder_mouth(correct_sample_mouth)


                image_name = path[0].split('/')[-1].replace('.png','').replace('.jpg','')
                save_path = base_path + "/encode_tensor/" + image_name

                assert os.path.exists(save_path) == True

                decode_embed_feature4 = torch.load(save_path+"/left_eye")
                decode_embed_feature5 = torch.load(save_path+"/right_eye")
                decode_embed_feature_skin = torch.load(save_path+"/skin")
                decode_embed_feature_hair = torch.load(save_path+"/hair")
                decode_embed_feature_mouth = torch.load(save_path+"/mouth")

            left_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
            right_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
            mouth_tensor = torch.zeros(encode_label_feature.size()).cuda()
            for batch_index in range(0,label.size()[0]):
                try:
                    left_eye_tensor[batch_index,:,int(mask_list[batch_index][0]/4+0.5)-4:int(mask_list[batch_index][0]/4+0.5)+4,int(mask_list[batch_index][1]/4+0.5)-6:int(mask_list[batch_index][1]/4+0.5)+6] += decode_embed_feature4[batch_index]
                    right_eye_tensor[batch_index,:,int(mask_list[batch_index][2]/4+0.5)-4:int(mask_list[batch_index][2]/4+0.5)+4,int(mask_list[batch_index][3]/4+0.5)-6:int(mask_list[batch_index][3]/4+0.5)+6] += decode_embed_feature5[batch_index]
                    mouth_tensor[batch_index,:,int(mask_list[batch_index][4]/4+0.5)-10:int(mask_list[batch_index][4]/4+0.5)+10,int(mask_list[batch_index][5]/4+0.5)-18:int(mask_list[batch_index][5]/4+0.5)+18] += decode_embed_feature_mouth[batch_index]
                except:
                    print("wrong ! ")

            fake_style_face = self.netG.forward(torch.cat((encode_label_feature,left_eye_tensor,right_eye_tensor,decode_embed_feature_skin,decode_embed_feature_hair,mouth_tensor),1),type="image_G")


            reconstruct_fake_style = self.netG.forward(torch.cat((fake_style_face,mask_style_bg_feature),1),type="bg_decoder")

        return reconstruct_fake_style






    def inference_parsing(self, label, image):
        # Encode Inputs        
        input_label, _, real_image, _, _ = self.encode_input(Variable(label), None, Variable(image), infer=True)
    
        assert torch.__version__.startswith('0.4')
        with torch.no_grad():
            # first go through the encoder and decoder
            self.netP.eval()
            parsing_label_feature = self.netP(real_image)
            _,g_label = torch.max(parsing_label_feature,dim=1)
            # real_label = softmax2label(real_label_feature)

        return g_label


    def inference_parsing_filter(self, ori_label, image):
        # Encode Inputs        
        input_label, _, real_image, _, _ = self.encode_input(Variable(ori_label), None, Variable(image), infer=True)
    
        assert torch.__version__.startswith('0.4')
        with torch.no_grad():
            # first go through the encoder and decoder
            self.netP.eval()
            parsing_label_feature = self.netP(real_image)
            gt_label = torch.squeeze(ori_label.type(torch.cuda.LongTensor),1)
            loss_parsing = self.criterionCrossEntropy(parsing_label_feature,gt_label)

        return loss_parsing

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.netD2, 'D2', which_epoch, self.gpu_ids)
        self.save_network(self.net_encoder_skin, 'encoder_skin', which_epoch, self.gpu_ids)
        self.save_network(self.net_encoder_hair, 'encoder_hair', which_epoch, self.gpu_ids)
        self.save_network(self.net_encoder_left_eye, 'encoder_left_eye', which_epoch, self.gpu_ids)
        self.save_network(self.net_encoder_right_eye, 'encoder_right_eye', which_epoch, self.gpu_ids)
        self.save_network(self.net_encoder_mouth, 'encoder_mouth', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_skin, 'decoder_skin', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_hair, 'decoder_hair', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_left_eye, 'decoder_left_eye', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_right_eye, 'decoder_right_eye', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_mouth, 'decoder_mouth', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_skin_image, 'decoder_skin_image', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_hair_image, 'decoder_hair_image', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_left_eye_image, 'decoder_left_eye_image', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_right_eye_image, 'decoder_right_eye_image', which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder_mouth_image, 'decoder_mouth_image', which_epoch, self.gpu_ids)
        self.save_network(self.netP, 'P', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D2.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G_together.param_groups:
            param_group['lr'] = lr        
        # for param_group in self.optimizer_sample_net.param_groups:
        #     param_group['lr'] = lr
        # for param_group in self.optimizer_vae_net.param_groups:
        #    param_group['lr'] = lr
        # for param_group in self.optimizer_netP.param_groups:
        #    param_group['lr'] = lr            
        # for param_group in self.optimizer_vae_encoder.param_groups:
        #     param_group['lr'] = lr
        # for param_group in self.optimizer_mask_autoencoder.param_groups:
        #     param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHD_mask_Model):
    def forward(self, label, inst, image, mask_list, label2, mask2):
        # label, inst = inp
        if self.opt.multi_embed_test == True:
            return self.inference_multi_embed(label, inst, image, mask_list, label2, mask2)
        elif self.opt.test_parsing == False:
            return self.inference(label, inst, image, mask_list, label2, mask2)
        else:
            return self.inference_parsing(label,inst,image)


        
