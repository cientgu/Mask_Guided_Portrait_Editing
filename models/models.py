### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch

def create_model(opt):
    if opt.model == 'pix2pixHD':
        if opt.mask_model == True:
            from .pix2pixHD_mask_model import Pix2PixHD_mask_Model, InferenceModel
            if opt.isTrain:
                model = Pix2PixHD_mask_Model()
            else:
                model = InferenceModel()
        elif opt.dist_model == True:
            from .pix2pixHD_dist_model import Pix2PixHD_dist_Model, InferenceModel
            if opt.isTrain:
                model = Pix2PixHD_dist_Model()
            else:
                model = InferenceModel()
        elif opt.sample_noise == True:
            from .pix2pixHD_sample_model import Pix2PixHD_sample_Model, InferenceModel
            if opt.isTrain:
                model = Pix2PixHD_sample_Model()
            else:
                model = InferenceModel()
        elif opt.vae_encoder == True:
            from .pix2pixHD_embed_model import Pix2PixHD_embed_Model, InferenceModel
            if opt.isTrain:
                model = Pix2PixHD_embed_Model()
            else:
                model = InferenceModel()
        else:
            from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
            if opt.isTrain:
                model = Pix2PixHDModel()
            else:
                model = InferenceModel()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % model.name)

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids, dim=0)

    return model
