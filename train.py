# this is training code fix parsing net and training with mask model
# from train_fixP.py


### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable
import scipy.misc
import random

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

print("print the model to file ------------")
network_save_path = os.path.join(opt.checkpoints_dir, opt.name, "network_structure.txt")
with open(network_save_path, 'wt') as f:
    if model.module.name == "Pix2PixHD_embed_Model" or model.module.name == "Pix2PixHD_sample_Model" or model.module.name == "Pix2PixHD_dist_Model" or model.module.name == "Pix2PixHD_mask_Model":
        print("netG network is :\n", file=f)
        print(model.module.netG, file=f)
        print("netD network is :\n", file=f)
        print(model.module.netD, file=f)
        if opt.dist_model == True:
            print("net_decoder is :\n", file=f)
            print(model.module.net_decoder, file=f)
            print("netD2 network is :\n", file=f)
            print(model.module.netD2, file=f)
            print("netP is :\n", file=f)
            print(model.module.netP, file=f)
            print("net_encoder is :\n", file=f)
            print(model.module.net_encoder, file=f)
        if opt.mask_model == True:
            print("netD2 network is :\n", file=f)
            print(model.module.netD2, file=f)
            print("netP is :\n", file=f)
            print(model.module.netP, file=f)
            print("net_encoder_skin is :\n", file=f)
            print(model.module.net_encoder_skin, file=f)
            print("net_encoder_hair is :\n", file=f)
            print(model.module.net_encoder_hair, file=f)        
            print("net_encoder_left_eye is :\n", file=f)
            print(model.module.net_encoder_left_eye, file=f)
            print("net_encoder_right_eye is :\n", file=f)
            print(model.module.net_encoder_right_eye, file=f)
            print("net_encoder_mouth is :\n", file=f)
            print(model.module.net_encoder_mouth, file=f)

            print("net_decoder_skin is :\n", file=f)
            print(model.module.net_decoder_skin, file=f)
            print("net_decoder_hair is :\n", file=f)
            print(model.module.net_decoder_hair, file=f)        
            print("net_decoder_left_eye is :\n", file=f)
            print(model.module.net_decoder_left_eye, file=f)
            print("net_decoder_right_eye is :\n", file=f)
            print(model.module.net_decoder_right_eye, file=f)
            print("net_decoder_mouth is :\n", file=f)
            print(model.module.net_decoder_mouth, file=f)

            print("net_decoder_skin_image is :\n", file=f)
            print(model.module.net_decoder_skin_image, file=f)
            print("net_decoder_hair_image is :\n", file=f)
            print(model.module.net_decoder_hair_image, file=f)        
            print("net_decoder_left_eye_image is :\n", file=f)
            print(model.module.net_decoder_left_eye_image, file=f)
            print("net_decoder_right_eye_image is :\n", file=f)
            print(model.module.net_decoder_right_eye_image, file=f)            
            print("net_decoder_mouth_image is :\n", file=f)
            print(model.module.net_decoder_mouth_image, file=f)            

    else:
        print("some model name we don't know")

loss_mean_temp = dict()
loss_count = 0
loss_names = ['KL_embed', 'L2_mask_image', 'G_GAN','G_GAN_Feat','G_VGG','D_real','D_fake','L2_image','ParsingLoss','G2_GAN','D2_real','D2_fake']
for loss_name in loss_names:
    loss_mean_temp[loss_name] = 0

    
    
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    a_count = 0
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        # losses, generated, label_out = model(Variable(data['label']), Variable(data['inst']), Variable(data['image']), Variable(data['feat']), infer=save_fake, type="sample_net")
        # # losses, generated = model.module.forward_sample_net(Variable(data['label']), Variable(data['inst']), Variable(data['image']), Variable(data['feat']), infer=save_fake)
        # losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        # loss_dict = dict(zip(model.module.loss_names, losses))

        # loss_sample_net = loss_dict['L1_label'] + loss_dict['L1_vector'] + loss_dict['G_sample_GAN'] + loss_dict['G_MFM']
        # loss_D2 = (loss_dict['D2_fake'] + loss_dict['D2_real']) * 0.5
        
        # loss_temp1, loss_temp2, loss_temp3, loss_temp4, loss_temp5, loss_temp6 = loss_dict['L1_label'], loss_dict['L1_vector'], loss_dict['G_sample_GAN'], loss_dict['D2_fake'], loss_dict['D2_real'], loss_dict['G_MFM']

        # model.module.optimizer_sample_net.zero_grad()
        # loss_sample_net.backward(retain_graph=True)
        # model.module.optimizer_sample_net.step()

        # # update discriminator weights
        # model.module.optimizer_D2.zero_grad()
        # loss_D2.backward()
        # model.module.optimizer_D2.step()

        if opt.debug_mask_part == True:
            losses, reconstruct, left_eye_reconstruct, right_eye_reconstruct, skin_reconstruct, hair_reconstruct, mouth_reconstruct, transfer_image, transfer_label = model( Variable(data['bg_image']), Variable(data['label']), Variable(data['inst']), Variable(data['image']), Variable(data['feat']), Variable(data['image_affine']), Variable(data['mask']), Variable(data['ori_label']), infer=save_fake, type="vae_net")
        else:
            losses, reconstruct, real_parsing_label = model(Variable(data['label']), Variable(data['inst']), Variable(data['image']), Variable(data['feat']), Variable(data['image_affine']), infer=save_fake, type="vae_net")

        # losses, reconstruct = model.module.forward_vae_net(Variable(data['label']), Variable(data['inst']), Variable(data['image']), Variable(data['feat']), infer=save_fake)
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        a = random.random()
        # loss_kl = loss_dict['KL_embed']*1000
        # loss_mask = loss_dict['L2_mask_image'] * 500
        # loss_vae_net = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)*100 + loss_dict['L2_image']*500
        
        if a_count == 1:
            a_count = 0
            a_weight = 0
        else:
            a_count = 1
            a_weight = 1
            
            
        loss_D2 = (loss_dict['D2_fake'] + loss_dict['D2_real']) * 0.5
        loss_G_together = loss_dict['G_GAN']*a_weight + loss_dict['G2_GAN'] + loss_dict['G_GAN_Feat']*a_weight + loss_dict['G_VGG']*1*a_weight + loss_dict['L2_image']*2*a_weight + loss_dict['L2_mask_image'] * 500 + loss_dict['ParsingLoss']*10
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 * a_weight
        
        model.module.optimizer_G_together.zero_grad()
        loss_G_together.backward()
        model.module.optimizer_G_together.step()

        model.module.optimizer_D.zero_grad()
        # update discriminator weights
        loss_D.backward()
        model.module.optimizer_D.step()
    
        # update discriminator weights
        model.module.optimizer_D2.zero_grad()
        loss_D2.backward()
        model.module.optimizer_D2.step()
            

        # loss_dict['L1_label'], loss_dict['L1_vector'], loss_dict['G_sample_GAN'], loss_dict['D2_fake'], loss_dict['D2_real'], loss_dict['G_MFM'] = loss_temp1, loss_temp2, loss_temp3, loss_temp4, loss_temp5, loss_temp6

        #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        #debug to save input if loss_kl is too large:
        # if loss_dict['KL_embed'].cpu().data.numpy() > 10 and epoch > 1:
        #     print("get wrong image batch !!")
        #     for index in range(0,data['label'].size()[0]):
        #         label = util.tensor2label(data['label'][index],11)
        #         label = scipy.misc.toimage(label)
        #         label.save("debug/"+str(epoch)+"_"+str(index)+"label.jpg")
        #         image = util.tensor2im(data['image'][index])
        #         image = scipy.misc.toimage(image)
        #         image.save("debug/"+str(epoch)+"_"+str(index)+".jpg")
        #     print("save wrong image over -- ")

        # save losses to loss_mean_temp
        for loss_name in loss_names:
            loss_mean_temp[loss_name] += loss_dict[loss_name].cpu().data.numpy()
            loss_count += 1

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            for loss_name in loss_names:
                loss_mean_temp[loss_name] = loss_mean_temp[loss_name].item() / loss_count

            # errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
            errors = {k: v for k, v in loss_mean_temp.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            for loss_name in loss_names:
                loss_mean_temp[loss_name] = 0
            loss_count = 0

        ### display output images
        if save_fake:
            if opt.debug_mask_part == True:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                       ('input_ori_label', util.tensor2label(data['ori_label'][0], opt.label_nc)),
                                       ('transfer_label', util.tensor2label(transfer_label.data[0], opt.label_nc)),
                                       ('transfer_image', util.tensor2im(transfer_image.data[0])),
                                       ('reconstruct_image', util.tensor2im(reconstruct.data[0])),
                                       ('real_image', util.tensor2im(data['bg_image'][0]))
                                       # ('parsing_label', util.tensor2label(label_out.data[0], opt.label_nc)),
                                       # ('real_parsing_label', util.tensor2label(real_parsing_label.data[0], opt.label_nc)),
                                       # ('reconstruct_left_eye', util.tensor2im(left_eye_reconstruct.data[0])),
                                       # ('reconstruct_right_eye', util.tensor2im(right_eye_reconstruct.data[0])),
                                       # ('reconstruct_skin', util.tensor2im(skin_reconstruct.data[0])),
                                       # ('reconstruct_hair', util.tensor2im(hair_reconstruct.data[0])),
                                       # ('reconstruct_mouth', util.tensor2im(mouth_reconstruct.data[0])),
                                       # ('mask_lefteye', util.tensor2im(left_eye_real.data[0]))
                                       ])
            else:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                       # ('generated_image', util.tensor2im(generated.data[0])),
                                       ('reconstruct_image', util.tensor2im(reconstruct.data[0])),
                                       ('real_image', util.tensor2im(data['image'][0])),
                                       # ('parsing_label', util.tensor2label(label_out.data[0], opt.label_nc)),
                                       ('real_parsing_label', util.tensor2label(real_parsing_label.data[0], opt.label_nc))
                                       ])

            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
     #   if total_steps % opt.save_latest_freq == save_delta:
      #      print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
       #     model.module.save('latest')            
        #    np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % 1 == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')   


    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
